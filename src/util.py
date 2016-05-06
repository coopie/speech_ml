import os
import errno
import yaml

# echoes the behaviour of mkdir -p
# from http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
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

def ttv_yaml_to_dict(path):
    with open(path, 'r') as f:
        return yaml.load(f.read())
        

def get_emotion_from_filename(filename):
    return filename.split('/')[-1].split('.')[0].split('_')[1]


def get_emotion_number_from_filename(filename):
    return EMOTION_NUMBERS[filename.split('/')[-1].split('.')[0].split('_')[1]]
