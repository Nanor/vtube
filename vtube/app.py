from PIL import Image
import cv2
import numpy as np

from .Avatar import Avatar
from .Pose import Pose
from .FaceLandmark import FaceLandmark
from .IrisLandmark import IrisLandmark
from .PoseNet import PoseNet


def main():
    cap = cv2.VideoCapture(0)

    flip = True
    webcam = True
    greenscreen = False

    debug_mode = 6
    debug_modes = 6

    group_index = None

    posenet = PoseNet()
    face_landmark = FaceLandmark()
    iris_landmark = IrisLandmark()
    pose = Pose(posenet, face_landmark, iris_landmark)

    face_bounds = None
    avatar = Avatar(pose, "avatar", flip)
    # avatar = Avatar(pose, "avatar_2", (-100, -100), 900, flip)

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

        if face_bounds is None or debug_mode in [1, 2]:
            posenet.update(frame)
            face_frame, face_bounds = posenet.extract_face(frame)

        face_landmark.update(frame, face_bounds)
        iris_landmark.update(frame, face_landmark.eye_bounds())
        pose.calc()

        output = avatar.draw(greenscreen).resize((512, 512), resample=Image.BICUBIC)
        avatar_frame = np.array(output)

        if webcam:
            if debug_mode == 1:
                posenet.draw(frame, False)
            elif debug_mode == 2:
                posenet.draw_face_bounds(frame)
            elif debug_mode == 3:
                face_landmark.draw(frame, group_index)
            elif debug_mode == 4:
                face_landmark.draw_eye_bounds(frame)
            elif debug_mode == 5:
                iris_landmark.draw(frame)
            elif debug_mode == 6:
                pose.debug(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("webcam", frame)

        avatar_frame = cv2.cvtColor(avatar_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", avatar_frame)

        # cv2.imshow('svg', svg())

        key = cv2.waitKey(1)
        if key & 0xFF == ord(" "):
            pose.reset()

        if key & 0xFF == ord("n"):
            debug_mode += 1
            if debug_mode > debug_modes:
                debug_mode = 0

        if key & 0xFF == ord("m"):
            if group_index is None:
                group_index = 0
            else:
                group_index += 1

        if key & 0xFF == ord("g"):
            greenscreen = not greenscreen

        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
