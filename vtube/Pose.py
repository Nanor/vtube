import cv2
from numpy.lib.twodim_base import eye
from collections import defaultdict

from .Vector import Vector


class Pose:
    def __init__(self, posenet, face, iris):
        self.posenet = posenet
        self.face = face
        self.iris = iris

        self.offsets = defaultdict(lambda: 0)

    def calc(self):
        # mouth_width = self.face.get("mouth_right")[0] - self.face.get("mouth_left")[0]
        # mouth_height = self.face.get("mouth_bottom")[1] - self.face.get("mouth_top")[1]
        # mouth_ratio = mouth_height / mouth_width
        # mouth_open = max(0.1, (mouth_ratio - 0.2) / 0.4)

        left_eye = Vector(*self.face.point("left_eye"))
        right_eye = Vector(*self.face.point("right_eye"))

        eye_angle = left_eye.angle(right_eye)
        eye_middle = (left_eye + right_eye) * 0.5
        nose = Vector(*self.face.point("nose"))

        params = {
            "root_angle": eye_angle.z * 0.4,
            "head_tilt": eye_angle.z,
            "head_look": eye_angle.y,
            "head_nod": (eye_middle.angle(nose).x + 15) * 0.5,
            "mouth_open": 0.1,
        }

        self.params = params
        # self.params = {
        #     k: v - self.offsets[k] if k != "mouth_open" else v
        #     for k, v in params.items()
        # }

    def reset(self):
        self.offsets = {k: self.offsets[k] + self.params[k] for k in self.params}

    def debug(self, frame):

        y = 50

        for (k, v) in self.params.items():
            cv2.putText(
                frame,
                "{} = {}".format(k, str(v)),
                (500, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 150, 150),
                1,
                cv2.LINE_AA,
            )

            y += 15

