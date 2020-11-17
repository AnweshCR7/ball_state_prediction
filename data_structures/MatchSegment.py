"""Import this class via utils.data_models"""
from utils.enums import TeamSide


class MatchSegment:
    frames = list()
    match_name = ""
    mirror_data = False

    def __init__(self, frames, match_name):
        self.frames = frames
        self.match_name = match_name
        self.attacking_team_side, self.defending_team_side = self.determine_attack_defense()
        self.flipped = False

    def Length(self):
        return len(self.frames)

    def determine_attack_defense(self):
        first_frame = self.frames[0]
        attacking_team = first_frame.ball_possession_team
        defending_team = TeamSide.Away if attacking_team == TeamSide.Home else TeamSide.Home
        return attacking_team, defending_team

    def flip_segment(self):
        self.flipped = True
        for frame in self.frames:
            frame.flip()

    def get_visualization_info(self):
        return [frame.get_visualization_info() for frame in self.frames]

    def get_defending_player_trajectories(self):
        first_frame = self.frames[0]
        player_trajectories = []
        for p in first_frame.persons:
            if p.team_side == self.defending_team_side:
                player_trajectories.append(
                    [next((x.pos for x in frame.persons if x.id == p.id), None) for frame in self.frames])
        return player_trajectories

    def get_attacking_player_trajectories(self):
        first_frame = self.frames[0]
        player_trajectories = []
        for p in first_frame.persons:
            if p.team_side == self.attacking_team_side:
                player_trajectories.append(
                    [next((x.pos for x in frame.persons if x.id == p.id), None) for frame in self.frames])
        return player_trajectories

    def get_ball_trajectory(self):
        ball_trajectory = [frame.ball.pos for frame in self.frames]
        return ball_trajectory

    # Get all players from attack or defense in a frame.
    def get_team_in_frame(self, frame_nr, side):
        frame = self.frames[frame_nr]
        if side == "def":
            team = frame.team_home_players if self.defending_team_side == TeamSide.Home else frame.team_away_players
        elif side == "att":
            team = frame.team_home_players if self.attacking_team_side == TeamSide.Home else frame.team_away_players
        else:
            print("Side {0} is unknown. Choose 'att' or 'def'.")
            return None
        return team

    # Get all player positions from attack or defense in a frame.
    def get_team_positions_in_frame(self, frame_nr, side):
        team = self.get_team_in_frame(frame_nr, side)
        if team is not None:
            player_positions = [p.pos for p in team]
            return player_positions
        else:
            return None

