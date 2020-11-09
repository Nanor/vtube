import cv2
import numpy as np

from .PoseNet import PoseNet
from .Pose import Pose
from .Face import Face
from .Avatar import Avatar
from .ImageUtils import extract


def main():
    flip = True
    webcam = False
    greenscreen = False

    h_index = 0

    posenet = PoseNet()
    face = Face()
    pose = Pose(posenet, face)

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
        face_frame, face_bounds = posenet.extract_face(frame)
        face.update(face_frame)

        pose.calc()

        avatar_frame = np.array(avatar.draw(greenscreen))

        if webcam:
            # (h, w, _) = face.shape
            # frame[:h, :w] = face

            # posenet.draw_face_bounds(frame)
            face.draw(extract(frame, face_bounds, True), h_index)

            pose.debug(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("webcam", frame)

        avatar_frame = cv2.cvtColor(avatar_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", avatar_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord(" "):
            pose.reset()

        if key & 0xFF == ord("n"):
            h_index += 1

        if key & 0xFF == ord("g"):
            greenscreen = not greenscreen

        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
