import numpy as np
import os
import errno
import yaml



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


def numpy_string_to_array(numpy_str):
    return np.array(numpy_str[1:-1].split(), dtype=float)
