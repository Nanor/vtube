from numpy.core.numerictypes import _construct_lookups
import cv2
import numpy as np
import tensorflow as tf

from .ImageUtils import extract


class PoseNet:
    def __init__(self, smoothing=0.7):
        self._interpreter = tf.lite.Interpreter(
            model_path="vtube/resources/models/posenet.tflite"
        )
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self.smoothing = smoothing
        self.pose = np.array([[0, 0, 0] for _ in range(17)])

    def detect(self, image):
        model_width = self._input_details[0]["shape"][1]
        model_height = self._input_details[0]["shape"][2]

        resized = cv2.resize(image, (model_width, model_height),)

        reshaped = (
            (
                np.array(resized)
                .astype(np.float32)
                .reshape(self._input_details[0]["shape"])
            )
            - 127.5
        ) / 127.5

        self._interpreter.set_tensor(self._input_details[0]["index"], reshaped)
        self._interpreter.invoke()

        output_data = self._interpreter.get_tensor(self._output_details[0]["index"])
        offset_data = self._interpreter.get_tensor(self._output_details[1]["index"])

        template_heatmaps = np.squeeze(output_data)
        template_offsets = np.squeeze(offset_data)

        self.template_heatmaps = template_heatmaps
        self.template_offsets = template_offsets

        template_show = np.squeeze((reshaped.copy() * 127.5 + 127.5) / 255.0)
        template_show = np.array(template_show * 255, np.uint8)
        template_kps = parse_output(template_heatmaps, template_offsets, 0.3)

        return np.array(template_kps) / [model_height, model_width, 1]

    def update(self, image):
        new_pose = self.detect(image)

        self.pose = [
            p * [1, 1, 0.5]
            if n_p[2] <= 0.1
            else n_p
            if p[2] <= 0.1
            else p * (self.smoothing) + n_p * (1 - self.smoothing)
            for p, n_p in zip(self.pose, new_pose)
        ]

        return self

    def draw(self, image, draw_labels=True):
        (height, width, _) = image.shape

        for i, [y, x, c] in enumerate(self.pose):
            if c > 0.1:
                coord = (int(x * width), int(y * height))

                cv2.circle(image, coord, 2, (0, 255, 255), -1)
                if draw_labels:
                    cv2.putText(
                        image,
                        labels[i],
                        coord,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        return image

    def get(self, label):
        index = labels.index(label)
        point = self.pose[index]
        return (point[1], point[0]) if point[2] > 0.1 else None

    def face_bounds(self, aspect=1):
        points = [self.get(l) for l in labels[:5]]

        (x, y) = self.get("nose")

        min_x = min(p[0] for p in points if p is not None)
        max_x = max(p[0] for p in points if p is not None)
        width = max_x - min_x

        bounds_scale = 0.7
        bounds = [
            x - width * bounds_scale,
            y - width * bounds_scale * aspect,
            x + width * bounds_scale,
            y + width * bounds_scale * aspect,
        ]

        return bounds

    def draw_face_bounds(self, frame):
        (h, w, _) = frame.shape
        face_bounds = np.array(self.face_bounds(w / h)) * [w, h, w, h]
        cv2.rectangle(
            frame,
            (
                int(face_bounds[0]),
                int(face_bounds[1]),
                int(face_bounds[2] - face_bounds[0]),
                int(face_bounds[3] - face_bounds[1]),
            ),
            (100, 100, 200),
            2,
        )

    def extract_face(self, frame):
        (h, w, _) = frame.shape
        b = self.face_bounds(w / h)

        return (extract(frame, b, True), b)


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
