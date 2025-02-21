import cv2
import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "raw_input")
            input_data = input_tensor.as_numpy()

            input_size = pb_utils.get_input_tensor_by_name(request, "input_size")
            input_size = input_size.as_numpy()

            img_mean = pb_utils.get_input_tensor_by_name(request, "img_mean")
            img_mean = img_mean.as_numpy()

            img_std = pb_utils.get_input_tensor_by_name(request, "img_std")
            img_std = img_std.as_numpy()

            tensors = []
            for image, size, mean, std in zip(
                input_data, input_size, img_mean, img_std
            ):
                processed_data = cv2.resize(image, size)
                processed_data = cv2.cvtColor(processed_data, cv2.COLOR_BGR2RGB)
                processed_data = processed_data.astype(np.float32)
                processed_data /= 255.0
                processed_data -= mean
                processed_data /= std
                processed_data = np.transpose(processed_data, (2, 0, 1))

                tensors.append(processed_data)
            tensors = np.stack(tensors, axis=0)

            output_tensor = pb_utils.Tensor("tensor", tensors)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses
