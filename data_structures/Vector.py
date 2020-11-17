import numpy as np
from math import acos, pi


class Vector(np.ndarray):
    """
    n-dimensional point used for locations.
    inherits +, -, * (as dot-product)
    > p1 = Point([1, 2])
    > p2 = Point([4, 5])
    > p1 + p2
    Point([5, 7])
    See https://gist.github.com/eigencoder/c029d7557e1f0828aec5 for more usage.
    """
    def __new__(cls, input_array=(0, 0)):
        """
        :param cls:
        :param input_array: Defaults to 2d origin
        """
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @property
    def z(self):
        return self[1]

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def __iter__(self):
        for x in np.nditer(self):
            yield x.item()

    def __abs__(self):
        return Vector([abs(self.x), abs(self.z)])

    def distanceTo(self, other):
        """
        Both points must have the same dimensions
        :return: Euclidean distance
        """
        return np.linalg.norm(self - other)

    def magnitude(self):
        return np.linalg.norm(self)

    length = magnitude

    def magnitude_squared(self):
        return self.x ** 2 + \
               self.z ** 2

    # from euclid3 (pyeuclid/ https://github.com/ezag/pyeuclid)
    def normalize(self):
        d = self.magnitude()
        if d:
            self.x /= d
            self.z /= d
        return self

    # from euclid3 (pyeuclid/ https://github.com/ezag/pyeuclid)
    def normalized(self):
        d = self.magnitude()
        if d:
            return Vector(self.x / d,
                          self.z / d)
        return self.copy()

    def set(self, x, z):
        self[0] = x
        self[1] = z

    @staticmethod
    def angle(v1, v2):  # Daniel K copied this from a repository without checking its validity.
        return acos(np.dot(v1, v2) / (v1.length() * v2.length()))

    @staticmethod
    def angleDeg(v1, v2):
        return Vector.angle(v1, v2) * 180.0 / pi
