import math
import cv2


class Pose:
    def __init__(self, posenet):
        self.posenet = posenet

    def calc(self):
        shoulder_diff_x = (
            self.posenet.get("left_shoulder")[0] - self.posenet.get("right_shoulder")[0]
        )
        shoulder_diff_y = (
            self.posenet.get("left_shoulder")[1] - self.posenet.get("right_shoulder")[1]
        )

        self.root_angle = math.degrees(
            math.atan(shoulder_diff_y / shoulder_diff_x) if shoulder_diff_x != 0 else 0
        )

    def debug(self, frame):
        cv2.putText(
            frame,
            "root_angle = {}".format(self.root_angle),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

