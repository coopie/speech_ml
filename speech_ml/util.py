import os
import errno
import yaml
import numpy as np
from os.path import split as split_path

from .data_names import *


# echoes the behaviour of mkdir -p
# from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]


EMOTION_NUMBERS = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7,
}


def yaml_to_dict(path):
    """The naming of this is bad."""
    with open(path, 'r') as f:
        return yaml.load(f.read())


def save_to_yaml_file(filepath, o):
    with open(filepath, 'w') as f:
        yaml.dump(o, f, default_flow_style=False)



# TODO: move these to speech_ml_experiements, where they belong
def get_emotion_from_filename(filename):
    return split_path(filename)[-1].split('.')[0].split('_')[1]


def get_emotion_number_from_filename(filename):
    return EMOTION_NUMBERS[split_path(filename)[-1].split('.')[0].split('_')[1]]


def filename_to_category_vector(filename, category=None):
    emotion_number = get_emotion_number_from_filename(filename)
    if category is None:
        zeros = np.zeros(len(EMOTIONS), dtype='int16')
        zeros[emotion_number] = 1
        return zeros
    else:
        category_number = EMOTION_NUMBERS[category]
        zeros = np.zeros(2)
        # 0 index is negative, 1 index is positive
        zeros[int(category_number == emotion_number)] = 1
        return zeros
