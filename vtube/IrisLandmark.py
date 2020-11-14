from vtube.TfModel import TfModel
import cv2


class IrisLandmark:
    def __init__(self):
        self.model = TfModel("iris_landmark")

    def update(self, frame, bounds):
        self.brow_points = {}
        self.iris_points = {}

        for side in ['left', 'right']:
            [brow_points, iris_points] = self.model.run(frame, bounds[side])

            self.brow_points[side] = self.model.extract3d(frame, bounds[side], brow_points)
            self.iris_points[side] = self.model.extract3d(frame, bounds[side], iris_points)

    def draw(self, frame):
        for side in ['left', 'right']:
            for i, point in enumerate(self.brow_points[side]):
                x = int(point[0])
                y = int(point[1])

                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

            for i, point in enumerate(self.iris_points[side]):
                x = int(point[0])
                y = int(point[1])

                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)

