FROM nvcr.io/nvidia/tritonserver:24.10-py3

RUN apt update && apt install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python "numpy<2.0"

RUN rm -rf /var/lib/apt/lists/*