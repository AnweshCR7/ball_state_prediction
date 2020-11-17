"""Import this class via utils.data_models"""
# from euclid3 import Point2
from data_structures.Vector import Vector


class Ball:
    pos = Vector([0, 0])
    height = 0
    speed = 0

    def __init__(self, x_pos, y_pos, z_pos, speed):
        self.pos = Vector([int(x_pos) / 100, int(z_pos) / 100])
        self.height = int(y_pos) / 100
        self.speed = float(speed)

    def get_visualization_info(self):
        return self.pos.x, self.pos.z

