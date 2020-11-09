def extract(image, bounds, normalised=False):
    if normalised:
        (h, w, _) = image.shape
        return image[
            int(bounds[1] * h) : int(bounds[3] * h),
            int(bounds[0] * w) : int(bounds[2] * w),
        ]
    else:
        return image[bounds[1] : bounds[3], bounds[0] : bounds[2]]

