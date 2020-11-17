import glob
import os
import os.path as path
from tqdm import tqdm
import utils.data_models as models
import utils.enums as enums

frame_split_separator = ':'
person_split_separator = ';'
actor_split_separator = ','


# This function should be used to read a .dat file that contains a whole match
# For example: folder = os.getcwd() + "\\data-raw\\1061522.dat
def read_match_file(dat_file):
    # all_data = []
    data = get_match_segment_from_file(dat_file)
    # all_data.append(data)
    return data


# This function should be used to read a folder that contains the segments of a single match
# For example: folder = os.getcwd() + "\\data\\1061522_FEY-WIL\\
def read_match_folder(folder):
    all_data = []
    if not folder.endswith("\\"):
        folder += "\\"
    for fi in glob.glob(folder + "*.dat", recursive=False):
        data = get_match_segment_from_file(fi)
        all_data.append(data)
    return all_data


def get_match_segment_from_file(file_path):
    frames = list()
    head, tail = path.split(file_path)
    file_name = path.basename(tail.split('.')[0])

    with open(file_path) as file:
        for line in tqdm(file.readlines()):
            frames.append(convert_tracab_string_to_frame(line))
            # TODO: debug condition (remove later)
            # if len(frames) >= 100:
            #     break

    return models.MatchSegment(frames, file_name)


def convert_tracab_string_to_frame(tracab_string):
    split1 = tracab_string.split(frame_split_separator)
    frame_nr = split1[0]
    person_data = split1[1]
    ball_data = split1[2]

    persons = list()

    for person_string in list(filter(None, person_data.split(person_split_separator))):
        persons.append(extract_person_from_data(person_string))

    ball = extract_ball_from_data(ball_data)
    clicker_tags = extract_clicker_tags_from_data(ball_data)

    return models.Frame(frame_nr, persons, ball, clicker_tags[0], clicker_tags[1])


def extract_person_from_data(person_string):
    split = person_string.split(actor_split_separator)

    team_nr = split[0]
    tracking_id = split[1]
    shirt_nr = split[2]
    x_pos = split[3]
    z_pos = split[4]
    speed = split[5]

    return models.Player(team_nr, tracking_id, shirt_nr, x_pos, z_pos, speed)


def extract_ball_from_data(ball_string):
    split = ball_string.split(actor_split_separator)
    x_pos = split[0]
    z_pos = split[1]
    y_pos = split[2]
    speed = split[3]

    return models.Ball(x_pos, y_pos, z_pos, speed)


def extract_clicker_tags_from_data(ball_string):
    split = ball_string.split(actor_split_separator)

    if split[4] == "A":
        ball_possession_team = enums.TeamSide.Away
    else:
        ball_possession_team = enums.TeamSide.Home

    if split[5] == "Dead":
        ball_state = enums.BallState.Dead
    else:
        ball_state = enums.BallState.Alive

    return ball_possession_team, ball_state
