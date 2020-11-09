import json
from PIL import Image


class Avatar:
    def __init__(self, pose, name, root=(0, 0), size=700, flip=False):
        self.pose = pose
        self.name = name
        self.root = root
        self.size = size
        self.flip = flip

        with open("./vtube/resources/avatars/{}.json".format(name), mode="r") as f:
            self._json = json.load(f)

        self._parts = {}

        for part in self._json:
            image = None
            for p in self._json[part]["parts"]:
                new_p = Image.open("./vtube/resources/avatars/{}".format(p))
                if image is None:
                    image = new_p
                else:
                    image.paste(new_p, (0, 0), new_p)

            p_width, p_height = image.size
            scaled = image.resize(
                (int(self.size), int(self.size * p_height / p_width)), Image.BILINEAR
            ).convert("RGBA")

            try:
                json_origin = self._json[part]["origin"]
                scale_factor = self.size / p_width
                origin = (json_origin[0] * scale_factor, json_origin[1] * scale_factor)
            except KeyError:
                origin = scaled.size[0] / 2, scaled.size[1] / 2

            self._parts[part] = (scaled, origin)

    def draw(self, greenscreen=False):
        background = (0, 255, 0) if greenscreen else (0, 0, 0)
        image = Image.new("RGB", (self.size, self.size), background)

        for (name, (part, origin)) in self._parts.items():
            root_angle = (
                self.pose.params["root_angle"] * self._json["body"]["tiltScale"]
            )

            if name != "body":
                try:
                    horz = (
                        self.pose.params["head_look"]
                        * self._json[name]["paralax"]
                        * 400
                    )
                    vert = (
                        -self.pose.params["head_nod"]
                        * self._json[name]["paralax"]
                        * 400
                    )

                    part = part.transform(
                        part.size, Image.AFFINE, (1, 0, horz, 0, 1, vert)
                    )
                except KeyError:
                    pass

                part = part.rotate(
                    self.pose.params["head_tilt"] - root_angle,
                    center=self._parts["head"][1],
                )

            part = part.rotate(root_angle, center=self._parts["body"][1])

            if self.flip:
                part = part.transpose(Image.FLIP_LEFT_RIGHT)

            image.paste(part, self.root, part)

        return image

