from pathlib import Path
from argparse import ArgumentParser, Namespace

import json

import cv2
import numpy as np
import tritonclient.http as httpclient


def main(opt: Namespace) -> None:
    # For Batching, resize to (512, 512)
    images = np.stack(
        [
            cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (512, 512))
            for path in opt.images
        ],
        axis=0,
    )

    with open(opt.class_names, "r", -1, "utf-8") as file_in:
        class_names = np.array(json.load(file_in))

    client = httpclient.InferenceServerClient(url="0.0.0.0:8000")

    inputs = httpclient.InferInput("RAW_INPUT", images.shape, datatype="UINT8")
    inputs.set_data_from_numpy(images, binary_data=True)

    outputs = httpclient.InferRequestedOutput("SCORES", binary_data=True)
    results = client.infer(
        model_name=opt.model,
        inputs=[inputs],
        outputs=[outputs],
    )
    inference_output = results.as_numpy("SCORES").astype(np.float32)

    indexes = np.argsort(-inference_output, axis=-1)
    scores = np.take_along_axis(inference_output, indexes, axis=-1)

    if opt.topk < 1:
        print(f"`topk must be positive integer: provided {opt.topk}`")
    topk = max(1, opt.topk)

    print("=" * 50)
    for batch_idx, (topk_indexes, topk_scores) in enumerate(
        zip(indexes[..., :topk], scores[..., :topk])
    ):
        print(f"{opt.images[batch_idx]}")
        for idx, score in zip(topk_indexes, topk_scores):
            print(f"\t[{idx:4d}] {class_names[idx]:25s}: {score * 100:.2f}%")
        if batch_idx != len(opt.images) - 1:
            print("-" * 50)
    print("=" * 50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        choices=("mobilenetv2", "resnet50"),
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=(
            "./resources/images/golden_retrieval.jpg",
            "./resources/images/red_fox.jpg",
        ),
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
