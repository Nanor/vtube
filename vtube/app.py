import cv2
import numpy as np

from .PoseNet import PoseNet
from .Pose import Pose
from .Avatar import Avatar


def main():
    flip = True

    posenet = PoseNet()
    pose = Pose(posenet)
    avatar = Avatar(pose, "avatar", (0, 100), 700, flip)
    # avatar = Avatar(pose, "avatar_2", (-100, -100), 900, flip)

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("output", 100, 100)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flip:
            frame = cv2.flip(frame, 1)

        posenet.update(frame)
        # posenet.draw(frame, False)

        pose.calc()

        pose.debug(frame)

        frame = avatar.draw(frame)
        frame = np.array(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
