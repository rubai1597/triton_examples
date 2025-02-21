from pathlib import Path
from argparse import ArgumentParser, Namespace

import json

import cv2
import numpy as np
import tritonclient.http as httpclient


def main(opt: Namespace) -> None:
    img = cv2.imread(opt.image)
    with open(opt.class_names, "r", -1, "utf-8") as file_in:
        class_names = np.array(json.load(file_in))

    client = httpclient.InferenceServerClient(url="0.0.0.0:8000")

    inputs = httpclient.InferInput("RAW_INPUT", img.shape[:3], datatype="UINT8")
    inputs.set_data_from_numpy(img, binary_data=True)
    outputs = httpclient.InferRequestedOutput("SCORES", binary_data=True)

    results = client.infer(model_name="mobilenetv2", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("SCORES").astype(np.float32)

    indexes = np.argsort(-inference_output, axis=-1)
    scores = np.take_along_axis(inference_output, indexes, axis=-1)

    if opt.topk < 1:
        print(f"`topk must be positive integer: provided {opt.topk}`")
    topk = max(1, opt.topk)

    for idx, score in zip(indexes[:topk], scores[:topk]):
        print(f"[{idx:4d}] {class_names[idx]:25s}: {score * 100:.2f}%")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="./resources/images/golden_retrieval.jpg",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default="./resources/texts/imagenet_classes.json",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
    )
    main(parser.parse_args())
