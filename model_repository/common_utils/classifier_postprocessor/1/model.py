import numpy as np

import triton_python_backend_utils as pb_utils


def softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    max_value = np.max(arr, axis=axis, keepdims=True)
    arr -= max_value

    numerator = np.exp(arr - max_value)
    denominator = np.sum(numerator, axis=axis, keepdims=True)

    return numerator / denominator


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "model_output")
            input_data = input_tensor.as_numpy()

            scores = softmax(input_data, axis=-1)

            output_tensor = pb_utils.Tensor("scores", scores.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses
