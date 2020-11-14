from vtube.TfModel import TfModel
import cv2
import numpy as np

from .ImageUtils import extract


class PoseNet:
    def __init__(self):
        self.model = TfModel("posenet")

    def update(self, image):

        (heatmaps, offsets, _, _) = self.model.run(image)

        points = parse_output(heatmaps, offsets, 0.3)

        (h, w, _) = image.shape
        (m_w, m_h) = self.model.size()

        self.points = [[x * w / m_w, y * h / m_h, c] for [y, x, c] in points]

    def draw(self, image, draw_labels=True):
        for i, [x, y, c] in enumerate(self.points):
            if c > 0.1:
                coord = (int(x), int(y))

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
        point = self.points[index]
        return (point[0], point[1]) if point[2] > 0.1 else None

    def face_bounds(self):
        (x1, y1) = self.get("left_eye")
        (x2, y2) = self.get("right_eye")

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        width = max_x - min_x

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2 + width / 2

        bounds_scale = 1.3
        bounds = [
            x - width * bounds_scale,
            y - width * bounds_scale,
            x + width * bounds_scale,
            y + width * bounds_scale,
        ]

        return bounds

    def draw_face_bounds(self, frame):
        face_bounds = np.array(self.face_bounds())
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
        b = self.face_bounds()

        return (extract(frame, b), b)


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
