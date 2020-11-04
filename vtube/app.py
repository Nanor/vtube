import cv2
import numpy as np

from .PoseNet import PoseNet


def main():
    posenet = PoseNet()

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("output", 100, 100)

    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        posenet.update(frame)
        posenet.draw(frame)

        cv2.imshow("output", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
