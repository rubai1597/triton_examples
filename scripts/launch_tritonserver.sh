docker run --rm \
--gpus all \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
-v ${PWD}/model_repository:/models \
--name tritonserver \
nvcr.io/nvidia/tritonserver:24.10-py3-opencv \
tritonserver \
--model-repository=/models/classification/mobilenetv2 \
--model-repository=/models/classification/resnet50 \
--model-repository=/models/common_utils \
--model-control-mode=poll \
--repository-poll-secs=15
