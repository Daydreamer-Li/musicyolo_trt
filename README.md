# MusicYOLO_with_trt
相较于musicyolo而言，改进如下：
1. 将依赖于yolox的函数单独提取出来，不再需要yolox的框架；
2. 将模型转换为tensorrt模型，加速了模型计算
3. 将分片存储的算法改为滑动窗口，目前不需要对图片切割存储，在onset和offset精度上有一定影响，具体见下表：


|     | f1     | p     | r    | 
| -------- | -------- |-------- |-------- |
| Onset| 95.55 |95.22 | 95.93|
| Onset_trt | 95.36 | 94.48 |  96.41 |
| Offset | 97.80 | 97.46 |98.19 |
| Offset_trt | 96.60 | 95.70 | 97.68 |
| COnP | 84.76 |84.46 | 85.10 |
| COnP_trt | 84.41 |83.61 | 85.36|
| COnPOff | 83.33 | 83.04 |83.66|
| COnPOff_trt | 82.52| 81.75 | 83.41|



## Inference

Step1. generate result.
```shell
python predict_with_window.py --audiodir $SSVD_TEST_SET_PATH --savedir $SAVE_PATH --ext .flac -cf ./yolo.json  -c ./model/alexnet_trt.pth  --device gpu
```

Step1. caculate f1.
```shell
python caculate_f1.py --label_path $SSVD_TEST_SET_PATH --result_path $YOUR_SAVE_PATH
```
