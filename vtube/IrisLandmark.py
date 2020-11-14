from vtube.TfModel import TfModel
import cv2


class IrisLandmark:
    def __init__(self):
        self.model = TfModel("iris_landmark")

    def update(self, frame, bounds):
        [brow_points, iris_points] = self.model.run(frame, bounds)

        self.brow_points = self.model.extract3d(frame, bounds, brow_points)
        self.iris_points = self.model.extract3d(frame, bounds, iris_points)

    def draw(self, frame):
        for i, point in enumerate(self.brow_points):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

        for i, point in enumerate(self.iris_points):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)

