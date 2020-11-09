import cv2
import numpy as np

from .PoseNet import PoseNet
from .Pose import Pose
from .Avatar import Avatar


def main():
    flip = True
    webcam = False
    greenscreen = False

    posenet = PoseNet()
    pose = Pose(posenet)
    avatar = Avatar(pose, "avatar", (0, 100), 700, flip)
    # avatar = Avatar(pose, "avatar_2", (-100, -100), 900, flip)

    cap = cv2.VideoCapture(0)

    if webcam:
        cv2.namedWindow("webcam", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("webcam", 50, 100)

    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("output", 900, 150)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if flip:
            frame = cv2.flip(frame, 1)

        posenet.update(frame)
        # posenet.draw(frame, False)

        pose.calc()

        pose.debug(frame)

        avatar_frame = np.array(avatar.draw(greenscreen))

        if webcam:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("webcam", frame)

        avatar_frame = cv2.cvtColor(avatar_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", avatar_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord(" "):
            pose.reset()

        if key & 0xFF == ord("g"):
            greenscreen = not greenscreen

        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
