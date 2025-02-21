# 1. Requirements
## 1.1 Run a Script to Create a Docker Image
- For example codes, tritonserver would have opencv-python
- run `scripts/create_image.sh` to create a docker image we need

```shell
cd scripts
chmod 775 create_image.sh
./create_image.sh
```

## 1.2 Create ONNX Files
- This repository needs mobilenetv2 and resnet50 onnx files to run example codes
- You can download model files from [Google Drive](https://drive.google.com/drive/folders/1S8mvvEb22hB72jfEoSTln7ytc6HAgUw9?usp=sharing)
- Or you can create model files manually with PyTorch
```python
import torch
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
)

targets = (
    (resnet50, ResNet50_Weights, "resnet50.onnx"),
    (mobilenet_v2, MobileNet_V2_Weights, "mobilenetv2.onnx"),
)

for module, weights, fname in targets:
    model = module(weights=weights.IMAGENET1K_V2)
    torch.onnx.export(
        model,
        torch.randn((1, 3, 224, 224), dtype=torch.float32),
        fname,
        opset=17,
        input_names=["input"],
        output_names=["output"],
    )
```
- ONNX files should be locatted in `model_repository` as shown below
- The name of model must be `model.onnx`
```shell
tree ./model_repository

# model_repository/
# ├── classification
# │   ├── mobilenetv2
# │   │   ├── mobilenetv2
# │   │   │   ├── 1
# │   │   │   └── config.pbtxt
# │   │   ├── mobilenetv2_config
# │   │   │   ├── 1
# │   │   │   │   └── model.py
# │   │   │   └── config.pbtxt
# │   │   └── mobilenetv2_onnx
# │   │       ├── 1
# │   │       │   └── model.onnx
# │   │       └── config.pbtxt
# │   └── resnet50
# │       ├── resnet50
# │       │   ├── 1
# │       │   └── config.pbtxt
# │       ├── resnet50_config
# │       │   ├── 1
# │       │   │   └── model.py
# │       │   └── config.pbtxt
# │       └── resnet50_onnx
# │           ├── 1
# │           │   └── model.onnx
# │           └── config.pbtxt
# └── common_utils
#     ├── classifier_postprocessor
#     │   ├── 1
#     │   │   └── model.py
#     │   └── config.pbtxt
#     └── image_preprocessor
#         ├── 1
#         │   └── model.py
#         └── config.pbtxt
```

# 2. Test
## 2.1 Launch TritonServer
- First, You should deploy tritonserver in your machine.
- run `scripts/launch_tritonserver.sh` to create tritonserver docker container
```shell
./scripts/launch_tritonserver.sh
```

## 2.2 Run Example Code
- `inference.py` will use ensemble model which includes pre-process, inference, and post-process.
- For MobileNetv2, the ensemble structure is written in `./model_repository/classification/mobilenetv2/mobilenetv2/config.pbtxt`
- The model name is `mobilenetv2` and it use `ensemble` platform. And it's input name is `RAW_INPUT` and output name is `SCORES`
- If you want, you can change these names what you want.
```plaintext
1  name: "mobilenetv2"
2  platform: "ensemble"
3
4  input [
5    {
6      name: "RAW_INPUT"
7      data_type: TYPE_UINT8
8      dims: [ -1,-1,3 ]
9    }
10  ]
11  output [
12    {
13      name: "SCORES"
14      data_type: TYPE_FP32
15      dims: [ -1 ]
16    }
17  ]
18
```

- When you run `inference.py`, you can get results like below:
    <img src=resources/images/golden_retrieval.jpg alt="golden retrieval" width="400"/>
```shell
python inference.py
# [ 207] Golden Retriever         : 51.52%
# [ 208] Labrador Retriever       : 9.12%
# [ 222] Kuvasz                   : 5.25%
# [ 257] Pyrenean Mountain Dog    : 3.17%
# [ 176] Saluki                   : 1.05%
```