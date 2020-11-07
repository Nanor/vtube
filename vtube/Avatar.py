import json
from PIL.Image import BILINEAR
import cv2
from PIL import Image, ImageFilter


class Avatar:
    def __init__(self, name, root=(0, 0), size=700, flip=False):
        self.name = name
        self.root = root
        self.size = size

        with open("./vtube/resources/avatars/{}.json".format(name), mode="r") as f:
            self.json = json.load(f)

        self._parts = {}

        for part in self.json:
            image = None
            for p in self.json[part]["parts"]:
                new_p = Image.open("./vtube/resources/avatars/{}".format(p))
                if image is None:
                    image = new_p
                else:
                    image.paste(new_p, (0, 0), new_p)

            p_width, p_height = image.size
            scaled = image.resize(
                (int(self.size), int(self.size * p_height / p_width)), Image.BILINEAR
            ).convert("RGBA")

            if flip:
                scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)

            self._parts[part] = scaled

    def draw(self, frame):
        image = Image.fromarray(frame)

        for part in self._parts.values():
            image.paste(part, self.root, part)

        return image

