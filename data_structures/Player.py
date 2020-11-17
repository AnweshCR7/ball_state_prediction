"""Import this class via utils.data_models"""
import utils.enums as enums
from data_structures.Vector import Vector


class Player:
    team_side = None
    tracking_id = 0
    shirt_nr = 0
    id = 0

    pos = Vector([0, 0])
    speed_ms = 0
    abs_speed = Vector([0, 0])

    def __init__(self, team_nr, tracking_id, shirt_nr, x_pos, z_pos, speed):
        self.team_side = enums.TeamSide(int(team_nr))
        self.tracking_id = tracking_id
        self.shirt_nr = int(shirt_nr)
        self.id = int(team_nr) * 100 + int(shirt_nr)
        self.pos = Vector([int(x_pos) / 100, int(z_pos) / 100])
        self.speed_ms = float(speed)

    def get_visualization_info(self):
        return self.shirt_nr, self.pos.x, self.pos.z
