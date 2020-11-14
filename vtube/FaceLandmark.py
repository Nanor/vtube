from vtube.Download import download_model
import numpy as np
import cv2

from .TfModel import TfModel


class FaceLandmark:
    def __init__(self):
        self.model = TfModel("face_landmark")

        self.bounds = None

    def update(self, frame, initial_bounds):
        (h, w, _) = frame.shape

        if self.bounds is None:
            self.bounds = np.array(initial_bounds)

        (points, scalar) = self.model.run(frame, self.bounds)

        self.points = self.model.extract3d(frame, self.bounds, points)

        self.update_bounds(w, h)

    def update_bounds(self, w, h):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        bounds = [min(xs), min(ys), max(xs), max(ys)]

        buffer = 100
        buffered = np.array(bounds) + [-buffer, -buffer, buffer, buffer]
        self.bounds = [
            min(max(buffered[0], 0), w),
            min(max(buffered[1], 0), h),
            min(max(buffered[2], 0), w),
            min(max(buffered[3], 0), h),
        ]

    def draw(self, frame, numbers=False):
        for i, point in enumerate(self.points):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

            if numbers:
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 150, 150),
                    1,
                    cv2.LINE_AA,
                )

        cv2.rectangle(
            frame,
            (
                int(self.bounds[0]),
                int(self.bounds[1]),
                int(self.bounds[2] - self.bounds[0]),
                int(self.bounds[3] - self.bounds[1]),
            ),
            (100, 100, 200),
            2,
        )
