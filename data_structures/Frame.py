"""Import this class via utils.data_models"""
import utils.enums as enums


class Frame:
    frame_nr = 0

    persons = None
    ball = None
    team_home_players = None
    team_away_players = None
    ball_possession_team = None
    ball_state = None

    def __init__(self, frame_nr, persons, ball, ball_possession_team,  ball_state):
        self.frame_nr = frame_nr
        self.persons = sorted(persons, key=lambda x: x.id)
        self.team_home_players = [p for p in self.persons if p.team_side == enums.TeamSide.Home]
        self.team_away_players = [p for p in self.persons if p.team_side == enums.TeamSide.Away]
        self.ball = ball
        self.ball_possession_team = ball_possession_team
        self.ball_state = ball_state

    def flip(self):
        # self.ball.pos.set(self.ball.pos.x * -1, self.ball.pos.z)
        self.ball.pos *= -1
        for p in self.persons:
            p.pos *= -1

    def get_visualization_info(self):
        team_untracked = [p for p in self.persons if p.team_side == enums.TeamSide.Untracked]
        if team_untracked:
            print("Warning! Frame {0}: There are {1} untracked players".format(self.frame_nr, len(team_untracked)))
        return self.ball, self.ball_possession_team, self.team_home_players, self.team_away_players

    def to_string(self):
        frame_string = f'{self.frame_nr}:'
        for p in self.persons:
            frame_string += f'{p.team_side.value},{p.tracking_id},{p.shirt_nr},{int(p.pos[0] * 100)},{int(p.pos[1] * 100)},{p.speed_ms:.2f};'

        frame_string += f':{int(self.ball.pos[0]* 100)},{int(self.ball.pos[1]* 100)},{int(self.ball.height * 100)},{self.ball.speed:.2f},' \
                        f'{"A" if self.ball_possession_team == enums.TeamSide.Away else "H"},' \
                        f'{"Alive" if self.ball_state == enums.BallState.Alive else "Dead"};:'

        return frame_string

