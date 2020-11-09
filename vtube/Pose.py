import math
import cv2
from numpy.lib.twodim_base import eye
from collections import defaultdict


class Pose:
    def __init__(self, posenet, face):
        self.posenet = posenet
        self.face = face

        self.offsets = defaultdict(lambda: 0)

    def calc(self):
        try:
            shoulder_diff_x = (
                self.posenet.get("left_shoulder")[0]
                - self.posenet.get("right_shoulder")[0]
            )
            shoulder_diff_y = (
                self.posenet.get("left_shoulder")[1]
                - self.posenet.get("right_shoulder")[1]
            )
            root_angle = math.degrees(
                math.atan(shoulder_diff_y / shoulder_diff_x)
                if shoulder_diff_x != 0
                else 0
            )
        except TypeError:
            root_angle = 0

        eye_diff_x = self.posenet.get("left_eye")[0] - self.posenet.get("right_eye")[0]
        eye_diff_y = self.posenet.get("left_eye")[1] - self.posenet.get("right_eye")[1]

        head_tilt = math.degrees(
            math.atan(eye_diff_y / eye_diff_x) if eye_diff_x != 0 else 0
        )

        head_look = (
            self.posenet.get("left_eye")[0] - self.posenet.get("nose")[0]
        ) / eye_diff_x - 0.5

        head_nod = (
            (
                (
                    (self.posenet.get("left_eye")[1] + self.posenet.get("right_eye")[1])
                    / 2
                )
                - self.posenet.get("nose")[1]
            )
            / eye_diff_x
        ) - 0.7

        mouth_width = self.face.get("mouth_right")[0] - self.face.get("mouth_left")[0]
        mouth_height = self.face.get("mouth_bottom")[1] - self.face.get("mouth_top")[1]
        mouth_ratio = mouth_height / mouth_width
        mouth_open = max(0.1, (mouth_ratio - 0.2) / 0.4)

        params = {
            "root_angle": root_angle,
            "head_tilt": head_tilt,
            "head_look": head_look,
            "head_nod": head_nod,
            "mouth_open": mouth_open,
        }

        self.params = {
            k: v - self.offsets[k] if k != "mouth_open" else v
            for k, v in params.items()
        }

    def reset(self):
        self.offsets = {k: self.offsets[k] + self.params[k] for k in self.params}

    def debug(self, frame):

        y = 50

        for (k, v) in self.params.items():
            cv2.putText(
                frame,
                "{} = {}".format(k, v),
                (500, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 150, 150),
                1,
                cv2.LINE_AA,
            )

            y += 15

