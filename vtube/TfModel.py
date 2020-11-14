import numpy as np
import tensorflow as tf
import cv2

from .Download import download_model
from .ImageUtils import extract


class TfModel:
    def __init__(self, model_name):
        download_model(model_name)

        self._interpreter = tf.lite.Interpreter('data/{}.tflite'.format(model_name))
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def run(self, image, bounds=None):
        model_width = self._input_details[0]["shape"][1]
        model_height = self._input_details[0]["shape"][2]

        if bounds is not None:
            image = extract(image, bounds)

        image = cv2.resize(image, (model_width, model_height))

        image = (
            np.array(image).astype(np.float32).reshape(self._input_details[0]["shape"])
        ) / (255 / 2) - 0.5

        self._interpreter.set_tensor(self._input_details[0]["index"], image)
        self._interpreter.invoke()

        outputs = []

        for i, details in enumerate(self._output_details):
            output = np.squeeze(self._interpreter.get_tensor(details["index"]))

            outputs.append(output)

        return outputs

    def extract3d(self, frame, bounds, data):
        h, w, _ = frame.shape
        model_width, model_height = self.size()

        points = []

        for point in np.reshape(data, (-1, 3)):
            if bounds is not None:
                points.append(
                    (
                        point
                        * [
                            (bounds[2] - bounds[0]) / model_width,
                            (bounds[3] - bounds[1]) / model_height,
                            1,
                        ]
                    )
                    + [bounds[0], bounds[1], 0]
                )
            else:
                points.append(point * [w / model_width, h / model_height])

        return points

    def size(self):
        return (self._input_details[0]["shape"][1], self._input_details[0]["shape"][2])

