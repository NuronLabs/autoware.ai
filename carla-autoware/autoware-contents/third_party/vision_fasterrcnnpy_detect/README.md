# How to Install Pytorch FasterRCNN Object Detector 
**Updated as of 2021-08-29**

1.

Once compiled, run from the terminal, or launch from RunTimeManager:

```
% roslaunch vision_fasterrcnnpy_detect vision_fasterrcnnpy_detect pretrained_model_file:=/PATH/TO/model.pth
```
Remember to modify the launch file `vision_fasterrcnnpy_detect.launch` and point the network and pre-trained models to your paths.

## Launch file params

|Parameter| Type| Description|Default|
----------|-----|--------|---|
|`use_gpu`|*Bool* |Whether to use or not GPU acceleration.|`true`|
|`gpu_device_id`|*Integer*|ID of the GPU to be used.|`0`|
|`score_threshold`|*Double*|Value between 0 and 1. Defines the minimum score value to filter detections.|`0.5`|
|`pretrained_model_file`|*String*|Path to the prototxt file .|`path/to/models/model.pth`|
|`camera_id`|*String*|Camera ID to subscribe, i.e. `camera0`|`/`|
|`image_src`|*String*|Name of the image topic to subscribe to|`/image_raw`|

## Notes
In order to build the app, you need to provide the FindTorch cmake files.

```cmake -DCMAKE_PREFIX_PATH="/home/ubuntu/anaconda3/envs/pytorch_latest_p37/lib/python3.7/site-packages/torch/share/cmake/Torch" ..
```
