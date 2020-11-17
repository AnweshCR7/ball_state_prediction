import numpy as np
import pandas as pd

feature_columns=['ball_height', 'ball_speed', 'ball_pos_x', 'ball_pos_z', 'ball_team', 'ball_state']


# Check more features later.
def get_features_from_match_data(data):
    frames = data.frames
    frames_array = []
    # df = np.zeros((len(frames), 5))
    # target = np.zeros(len(frames))
    for frame in frames:
        frames_array.append([frame.ball.height, frame.ball.speed, frame.ball.pos.x, frame.ball.pos.z,
                   frame.ball_possession_team.value, frame.ball_state.value])
        # target[idx] = frame.ball_state.value

    df = pd.DataFrame(data=frames_array, columns=feature_columns)

    return df
