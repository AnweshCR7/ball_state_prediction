from enum import Enum


class TeamSide(Enum):
    Untracked = -1
    Away = 0
    Home = 1
    Referee = 4


class BallState(Enum):
    Dead = 0
    Alive = 1
