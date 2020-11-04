from numpy.core.numerictypes import _construct_lookups
import cv2
import numpy as np
import tensorflow as tf


class PoseNet:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(
            model_path="vtube/resources/models/posenet.tflite"
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image):
        (height, width, _) = image.shape

        model_width = self.input_details[0]["shape"][1]
        model_height = self.input_details[0]["shape"][2]

        resized = cv2.resize(
            image,
            (model_width, model_height),
        )

        reshaped = (
            (
                np.array(resized)
                .astype(np.float32)
                .reshape(self.input_details[0]["shape"])
            )
            - 127.5
        ) / 127.5

        self.interpreter.set_tensor(self.input_details[0]["index"], reshaped)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        offset_data = self.interpreter.get_tensor(self.output_details[1]["index"])

        template_heatmaps = np.squeeze(output_data)
        template_offsets = np.squeeze(offset_data)

        self.template_heatmaps = template_heatmaps
        self.template_offsets = template_offsets

        template_show = np.squeeze((reshaped.copy() * 127.5 + 127.5) / 255.0)
        template_show = np.array(template_show * 255, np.uint8)
        template_kps = parse_output(template_heatmaps, template_offsets, 0.3)

        image_coords = (
            template_kps * np.array([height / model_height, width / model_width, 1])
        ).astype(int)

        return {
            label: {
                "y": image_coords[i][0],
                "x": image_coords[i][1],
                "confidence": image_coords[i][2],
            }
            for (i, label) in enumerate(labels)
        }

    def update(self, image):
        self.pose = self.detect(image)

        return self

    def draw(self, image):
        for (label, part) in self.pose.items():
            x, y, confidence = (part["x"], part["y"], part["confidence"])

            if confidence:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                cv2.putText(
                    image,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return image


labels = [
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_foot",
    "left_foot",
]


def parse_output(heatmap_data, offset_data, threshold):

    """
    Input:
      heatmap_data - hetmaps for an image. Three dimension array
      offset_data - offset vectors for an image. Three dimension array
      threshold - probability threshold for the keypoints. Scalar value
    Output:
      array with coordinates of the keypoints and flags for those that have
      low probability
    """

    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(heatmap_data.shape[-1]):

        joint_heatmap = heatmap_data[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
        pose_kps[i, 0] = int(
            remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i]
        )
        pose_kps[i, 1] = int(
            remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num]
        )
        max_prob = np.max(joint_heatmap)

        if max_prob > threshold:
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps