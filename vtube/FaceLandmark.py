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

    def draw(self, frame, group_index=None):
        group = (
            list(MESH_ANNOTATIONS.values())[group_index % len(MESH_ANNOTATIONS)]
            if group_index is not None
            else None
        )

        for i, point in enumerate(self.points):
            x = int(point[0])
            y = int(point[1])

            if group is None or i in group:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

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

    def draw_eye_bounds(self, frame):
        bounds = self.eye_bounds()

        for b in bounds.values():
            cv2.rectangle(
                frame,
                (int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]),),
                (100, 100, 200),
                2,
            )

    def eye_bounds(self):
        bounds = {}
        for side in ["left", "right"]:
            groups = [k for k in MESH_ANNOTATIONS.keys() if "{}Eye".format(side) in k]
            indexes = set(i for g in groups for i in MESH_ANNOTATIONS[g])
            points = [self.points[i] for i in indexes]

            bounds[side] = [
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points),
            ]

        return bounds

    def point(self, label):
        if label == "nose":
            groups = ["noseTip"]
        elif label == "left_eye":
            groups = [k for k in MESH_ANNOTATIONS.keys() if "leftEye" in k]
        elif label == "right_eye":
            groups = [k for k in MESH_ANNOTATIONS.keys() if "rightEye" in k]
        else:
            raise Exception("Unknown label {}".format(label))

        indexes = set(i for g in groups for i in MESH_ANNOTATIONS[g])
        points = [self.points[i] for i in indexes]

        return np.sum(points, axis=0) / len(points)


MESH_ANNOTATIONS = {
    "silhouette": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "leftEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "leftEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "leftEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "leftEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "leftEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "leftEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "leftEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "leftEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "leftEyebrowLower": [35, 124, 46, 53, 52, 65],
    "rightEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "rightEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "rightEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "rightEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "rightEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "rightEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "rightEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "rightEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "rightEyebrowLower": [265, 353, 276, 283, 282, 295],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseLeftCorner": [98],
    "noseRightCorner": [327],
    "leftCheek": [205],
    "RightCheek": [425],
}

