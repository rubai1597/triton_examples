import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "raw_input")

            input_size = pb_utils.Tensor("input_size", np.array((224, 224), np.uint32))
            img_mean = pb_utils.Tensor(
                "img_mean",
                np.array((0.485, 0.456, 0.406), np.float32),
            )
            img_std = pb_utils.Tensor(
                "img_std",
                np.array((0.229, 0.224, 0.225), np.float32),
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[input_tensor, input_size, img_mean, img_std]
                )
            )

        return responses
