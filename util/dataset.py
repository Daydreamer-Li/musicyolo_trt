#本代码集成数据载入和音频数据转换为图像数据
import os
import cv2
import random
import numpy as np
import math
import torch
import soundfile as sf
import torch.nn as nn
from nnAudio import features
import librosa
class TestDataset():
    def __init__(self, cqt_config,  audio_path, aug=False):
        sr = cqt_config['sr']
        bins_per_octave = cqt_config['bins_per_octave']
        fmin = cqt_config['fmin']
        hop = cqt_config['hop']
        stride = hop / sr
        frame = int(cqt_config['duration'] / stride)

        n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1) # 19,855.76
        n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 32768 2.048s
        # print(n_fft)
        # n_fft = 90000
        window_length = (frame - 1) * hop + n_fft # 150 frame 5.028s
        overlap_ratio = cqt_config['overlap_ratio']
        big_hop = int(cqt_config['duration'] * sr * (1 - overlap_ratio))

        if audio_path.endswith('.npy'):
            audio = np.load(audio_path, allow_pickle=True).astype('float32') / 32768
        else:
            audio, sr = sf.read(audio_path, dtype='float32')
            assert sr == cqt_config['sr'], 'the audio should be resample to {} first'.format(cqt_config['sr'])
        
        audio = np.pad(audio, (n_fft//2, 0))
        
        num = int(np.ceil((len(audio) - window_length) / big_hop)) + 1
        expect_len = window_length + (num - 1) * big_hop
        audio = np.pad(audio, (0, expect_len - len(audio)))        
        
        self.audio = audio
        self.num = num
        self.aug = aug
        self.big_hop = big_hop
        self.window_length = window_length

    def __getitem__(self, index):

        start = index * self.big_hop
        stop = start + self.window_length
        feature = np.zeros((self.window_length, ), dtype='float32')
        feature[: self.window_length] = self.audio[start: stop]

        return torch.from_numpy(feature)
    
    def get_total_audio(self):
        return torch.from_numpy(self.audio)

    def __len__(self):
        return self.num
    
class CQTSpectrogram(nn.Module):
    def __init__(self, cqt_config, width, height, log_scale=True, interpolate=True, convert2image=False):
        super(CQTSpectrogram, self).__init__()
        self.cqt = features.cqt.CQT(sr=cqt_config['sr'], hop_length=cqt_config['hop'], 
                                            fmin=cqt_config['fmin'], n_bins=cqt_config['n_bins'],
                                            bins_per_octave=cqt_config['bins_per_octave'], 
                                            center=False)
        self.width = width
        self.height = height
        self.log_scale = log_scale
        self.interpolate = interpolate
        self.convert2image = convert2image
        
    def forward(self, audio):
        # [1, 1, 176, 150]
        cqt = self.cqt(audio)[:, None]

        
        device = cqt.device
        cqt = cqt.cpu().numpy()[:, 0]
        cqt = librosa.amplitude_to_db(np.abs(cqt))
        cqt = cqt - cqt.min(axis=(-1, -2), keepdims=True)
        cqt /= (cqt.max(axis=(-1, -2), keepdims=True) + 1e-6)
        cqt = (cqt * 255).astype('uint8')
        b, h, w = cqt.shape
        cqt = cqt.reshape(-1, w)
        cqt = cv2.applyColorMap(cqt, cv2.COLORMAP_VIRIDIS).astype('float32')
        cqt= cv2.flip(cqt, 0)
        # cqt = cqt.reshape(-1, h, w, 3).transpose(0, 3, 1, 2)
        # cqt = torch.from_numpy(cqt).to(device)

        return cqt