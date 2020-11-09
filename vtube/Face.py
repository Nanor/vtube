from os import error
import cv2
import os
import requests
import numpy as np

LBFmodel = "lbfmodel.yaml"
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"


class Face:
    def __init__(self, smoothing=0.4):
        self.smoothing = smoothing

        if not os.path.isdir("data"):
            os.mkdir("data")

        if LBFmodel not in os.listdir("data"):
            r = requests.get(LBFmodel_url)
            open("data/{}".format(LBFmodel), "wb").write(r.content)

        self.detector = cv2.face.createFacemarkLBF()
        self.detector.loadModel("data/{}".format(LBFmodel))

    def update(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        face_bounds = cv2.UMat(np.array([[0, 0, frame.shape[1], frame.shape[0]]]))

        _, [landmarks] = self.detector.fit(grey, face_bounds)

        new_landmarks = np.array(cv2.UMat.get(landmarks))[0]

        try:
            self.landmarks = [
                p * (self.smoothing) + n_p * (1 - self.smoothing)
                for p, n_p in zip(self.landmarks, new_landmarks)
            ]
        except AttributeError:
            self.landmarks = new_landmarks

    def draw(self, frame, h_index):
        cv2.putText(
            frame,
            "face_point = {}".format(h_index),
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 150, 150),
            1,
            cv2.LINE_AA,
        )

        for i, (x, y) in enumerate(self.landmarks):
            cv2.circle(
                frame, (x, y), 1, (0, 255, 0) if h_index == i else (255, 0, 0), 1,
            )

    def get(self, point):
        point_index = {
            "mouth_top": 51,
            "mouth_bottom": 57,
            "mouth_left": 48,
            "mouth_right": 54,
        }

        return self.landmarks[point_index[point]]
