#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
# import tensorrt as trt
import cv2
import json
import torch
import torchvision
import numpy as np
import util.dataset
import util.boxes_to_notes
import util.get_yolo_pitch
import util.post_process
from torch2trt import TRTModule
import soundfile as sf
import librosa

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("MusicYOLO")
    parser.add_argument("--audiodir", type=str)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--ext", type=str, default='.flac')
    parser.add_argument("--prefix", type=bool, default=False)
    parser.add_argument('--config', '-cf', type=str, default='', 
                        help='Specify config')

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--num_classes", default=1, type=int, help="detect the number of classes")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model_trt,
        args,
        device="cpu",
    ):
        self.model_trt = model_trt
        self.num_classes = args.num_classes
        self.confthre = args.conf
        self.nmsthre = args.nms
        self.test_size = args.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
    
    def preproc(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    

    def get_img(self, image_name):
        img_info = {"id": 0}
        if isinstance(image_name, str):
            img_info["file_name"] = os.path.basename(image_name)
            img = cv2.imread(image_name)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img, ratio = self.preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = np.expand_dims(img, axis=0)
        return img,img_info
    
    ##yolox中内置函数，用于处理outputs
    def decode_outputs(self,outputs, dtype):#代替self.hw和self.strides
        grids = []
        strides = []
        selfhw = [(80, 80), (40, 40), (20, 20)]
        selfstrides = [8, 16, 32]
        for (hsize, wsize), stride in zip(selfhw, selfstrides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        # outputs = torch.rand((1,84,2))
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def inference(self, img):
        img_info = {"id": 0}
        img, ratio = self.preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        # np.save('/home/data/liyuqing/MusicYOLO/tools/1.npy',img)
        img = torch.from_numpy(img).unsqueeze(0)
      
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            
            outputs_trt = self.model_trt(img)
           
            outputs_trt = self.decode_outputs(outputs_trt, dtype=outputs_trt.type())
           
            outputs_trt = util.post_process.postprocess(outputs_trt, self.num_classes, self.confthre, self.nmsthre)

        return outputs_trt, img_info

    def predict(self, output, img_info, save_dir, cls_conf=0.35):
        ratio = img_info["ratio"]
        file_name = img_info["file_name"]
        height, width = img_info["height"], img_info["width"]
        ext = os.path.splitext(file_name)[1]

        if output is None:
            jsonname = file_name.replace(ext, '.json')
            jsonpath = os.path.join(save_dir, jsonname)
            data = {"file_name": file_name,
                    "img_size": [height, width],
                    "boxs": []}
            with open(jsonpath, 'wt') as f:
                f.write(json.dumps(data))
            return

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        bboxs = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = scores[i]
            if score < cls_conf:
                continue
            bboxs.append([float(co) for co in box])

        jsonname = file_name.replace(ext, '.json')
        jsonpath = os.path.join(save_dir, jsonname)
        data = {"file_name": file_name,
                "img_size": [height, width],
                "boxs": bboxs}
        with open(jsonpath, 'wt') as f:
            f.write(json.dumps(data))


def main(args):

   
    res_dir = os.path.join(args.savedir, 'res')


    os.makedirs(res_dir, exist_ok=True)

   
    args.resdir = res_dir
    args.test_size = (args.tsize, args.tsize)

    model_trt = TRTModule().cuda()
    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file)
    model_trt.load_state_dict(ckpt)
    

   
    logger.info("loaded checkpoint done.")

    # generate bounding box
    predictor = Predictor(model_trt, args, args.device)

    logger.info("Args: {}".format(args))
    with open(args.config) as f:
        cfg = json.load(f)
    #一些参数初始化
    batch_size = 1
    filepaths = []
    for file in os.listdir(args.audiodir):
        if file.endswith(args.ext):
            filepaths.append(os.path.join(args.audiodir, file))
   
    if args.device == "gpu":
        cqt = util.dataset.CQTSpectrogram(cfg["cqt"], cfg["m_config"]["width"], 
                                          cfg["m_config"]["height"],  convert2image=cfg["m_config"]["convert2image"]).cuda()
    for fileno, filepath in enumerate(filepaths):
        basename = os.path.basename(filepath).split('.')[0]

        cfg["cqt"]["overlap_ratio"] = cfg["test"]["overlap_ratio"]
        testset = util.dataset.TestDataset(cfg["cqt"], filepath)
        loader = torch.utils.data.DataLoader(testset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                )
        output_boxes = []
        for index, data in enumerate(loader): # [6, 80448] 等效于多张频谱图
            if args.device == "gpu":
                data = data.to('cuda')
            data = cqt(data)
            output,img_info = predictor.inference(data)
            if output[0] == None:
                empty = []
                empty = torch.from_numpy(np.array(empty))
                output_boxes.append(empty)
                continue
    
            bboxes = output[0][:,:5]
        
            bboxes /= img_info['ratio']
            output_boxes.append(bboxes)
           
        scale_h = cfg["cqt"]["n_bins"] / (cfg["cqt"]["bins_per_octave"] / 12) / cfg["m_config"]["height"]
        scale_w = cfg["cqt"]["duration"] / cfg["m_config"]["width"]
        hop = (1. - cfg["cqt"]["overlap_ratio"]) * cfg["m_config"]["width"]
        total_notes = util.boxes_to_notes.convert_boxs_to_notes(output_boxes, hop, scale_h, scale_w, cfg["m_config"]["width"], cfg["m_config"]["height"])
        total_notes.sort(key=lambda x: x[0])
        
       
        total_notes = util.get_yolo_pitch.get_yolo_note_pitch(cfg, total_notes, filepath)
        print(filepath)
        
        savepath = os.path.join(args.resdir,basename+'.txt')
         
        with open(savepath, 'w') as f:
            for onset, offset, pitch in total_notes:
                f.write('{:.3f}\t{:.3f}\t{:.3f}\n'.format(onset, offset, pitch))
       

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)

