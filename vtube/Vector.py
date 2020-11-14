import math


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "<Vector x={0.x}, y={0.y}, z={0.z}>".format(self)

    def angle(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z

        return Vector(
            math.degrees(math.atan(dz / dy)),
            math.degrees(math.atan(dz / dx)),
            math.degrees(math.atan(dy / dx)),
        )

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)
