from vtube.TfModel import TfModel
import cv2
import numpy as np


class IrisLandmark:
    def __init__(self):
        self.model = TfModel("iris_landmark")

    def update(self, frame, bounds):
        self.brow_points = {}
        self.iris_points = {}

        for side in ["left", "right"]:
            [brow_points, iris_points] = self.model.run(frame, bounds[side])

            self.brow_points[side] = self.model.extract3d(
                frame, bounds[side], brow_points
            )
            self.iris_points[side] = self.model.extract3d(
                frame, bounds[side], iris_points
            )

    def point(self, label):
        if label == "left_iris":
            return self.iris_points["left"][0]
        if label == "right_iris":
            return self.iris_points["right"][0]

        points = []
        if label == "left_eye":
            points = self.brow_points["left"][:16]
        if label == "right_eye":
            points = self.brow_points["right"][:16]

        return np.sum(points, axis=0) / 16

    def bounds(self, label):
        points = []
        if label == "left_eye":
            points = self.brow_points["left"][:16]
        if label == "right_eye":
            points = self.brow_points["right"][:16]

        min_x = np.array(points)[:, 0].min()
        min_y = np.array(points)[:, 1].min()
        max_x = np.array(points)[:, 0].max()
        max_y = np.array(points)[:, 1].max()

        return [min_x, min_y, max_x, max_y]

    def draw(self, frame):
        for side in ["left", "right"]:
            for i, point in enumerate(self.brow_points[side][:16]):
                x = int(point[0])
                y = int(point[1])

                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

            for i, point in enumerate(self.iris_points[side]):
                x = int(point[0])
                y = int(point[1])

                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)

