def extract(image, bounds):
    (h, w, _) = image.shape
    return image[
        max(0, int(bounds[1])) : min(h, int(bounds[3])),
        max(0, int(bounds[0])) : min(w, int(bounds[2])),
    ]

