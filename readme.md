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
- Notice that a model should have dynamic axes on batch if you want to use `dynamic batching`
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
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
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

## 1.3 Dynamic Batching
- If you set dynamic_batching on, tritonserver can handle large number of requests with dividing into batches. Otherwise, it is processed as like batch size is 1
- [See more about in here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- You can add `dynamic_batching` in config files except ensemble model.
- When `dynamic_batching` is on, `max_batch_size` should set to be larger than `0`
- Notice that `dynamic_batching` is not compatible with ensemble model. An ensemble model is handled with ensemble_scheduling. You just add `dynamic_batching` to each model used in the ensemble model.
```plaintext
 1  name: "mobilenetv2_config"
 2  backend: "python"
 3
 4  max_batch_size: 8
 5  instance_group [
 6    {
 7      kind: KIND_CPU
 8    }
 9  ]
10  dynamic_batching {
11    preferred_batch_size: [ 4, 8 ]
12    max_queue_delay_microseconds: 100
13  }
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
 4  max_batch_size: 8
 5
 6  input [
 7    {
 8      name: "RAW_INPUT"
 9      data_type: TYPE_UINT8
10      dims: [ -1,-1,3 ]
11    }
12  ]
13  output [
14    {
15      name: "SCORES"
16      data_type: TYPE_FP32
17      dims: [ -1 ]
18    }
19  ]
```

- When you run `inference.py`, you can get results like below:
    <img src=resources/images/golden_retrieval.jpg alt="golden retrieval" width="400"/>
    <img src=resources/images/red_fox.jpg alt="golden retrieval" width="400"/>
```shell
python inference.py --model mobilenetv2
# ==================================================
# ./resources/images/golden_retrieval.jpg
#         [ 207] Golden Retriever         : 47.27%
#         [ 208] Labrador Retriever       : 7.38%
#         [ 222] Kuvasz                   : 4.41%
#         [ 257] Pyrenean Mountain Dog    : 3.78%
#         [ 176] Saluki                   : 1.10%
# --------------------------------------------------
# ./resources/images/red_fox.jpg
#         [ 277] red fox                  : 12.15%
#         [ 272] coyote                   : 7.00%
#         [ 278] kit fox                  : 5.79%
#         [ 274] dhole                    : 4.90%
#         [ 269] grey wolf                : 1.49%
# ==================================================
```