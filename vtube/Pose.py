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

        self.params = {}

    def calc(self):
        # mouth_width = self.face.get("mouth_right")[0] - self.face.get("mouth_left")[0]
        # mouth_height = self.face.get("mouth_bottom")[1] - self.face.get("mouth_top")[1]
        # mouth_ratio = mouth_height / mouth_width
        # mouth_open = max(0.1, (mouth_ratio - 0.2) / 0.4)

        left_eye = Vector(*self.face.point("left_eye"))
        right_eye = Vector(*self.face.point("right_eye"))

        left_iris = Vector(*self.iris.point("left_iris"))
        right_iris = Vector(*self.iris.point("right_iris"))

        left_eye_bounds = self.iris.bounds('left_eye')
        right_eye_bounds = self.iris.bounds('right_eye')

        left_eye_x = (left_iris.x - left_eye_bounds[0]) / (left_eye_bounds[2] - left_eye_bounds[0])
        left_eye_y = (left_iris.y - left_eye_bounds[1]) / (left_eye_bounds[3] - left_eye_bounds[1])

        right_eye_x = (right_iris.x - right_eye_bounds[0]) / (right_eye_bounds[2] - right_eye_bounds[0])
        right_eye_y = (right_iris.y - right_eye_bounds[1]) / (right_eye_bounds[3] - right_eye_bounds[1])

        eye_angle = left_eye.angle(right_eye)
        eye_middle = (left_eye + right_eye) * 0.5
        nose = Vector(*self.face.point("nose"))

        params = {
            "root_angle": eye_angle.z * 0.4,
            "head_tilt": eye_angle.z,
            "head_look": eye_angle.y,
            "head_nod": (eye_middle.angle(nose).x + 15) * 0.5,
            "mouth_open": 0.1,
            "left_eye_x": (left_eye_x - 0.5) * 20,
            "left_eye_y": (left_eye_y - 0.5) * 30,
            "right_eye_x": (right_eye_x - 0.5) * 20,
            "right_eye_y": (right_eye_y - 0.5) * 30,
        }

        self.params = params

    def reset(self):
        self.offsets = {k: self.offsets[k] + self.params[k] for k in self.params}

    def debug(self, frame):

        y = 50

        for (k, v) in self.params.items():
            cv2.putText(
                frame,
                "{} = {}".format(k, str(v)),
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "{} = {}".format(k, str(v)),
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 150, 150),
                1,
                cv2.LINE_AA,
            )

            y += 15

