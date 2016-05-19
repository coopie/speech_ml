import os
import errno
import yaml
import h5py
import numpy as np
from data_names import *


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

def ttv_yaml_to_dict(path):
    with open(path, 'r') as f:
        return yaml.load(f.read())


def save_to_yaml_file(filepath, o):
    with open(filepath, 'w') as f:
        yaml.dump(o, f, default_flow_style=False)


def get_emotion_from_filename(filename):
    return filename.split('/')[-1].split('.')[0].split('_')[1]


def get_emotion_number_from_filename(filename):
    return EMOTION_NUMBERS[filename.split('/')[-1].split('.')[0].split('_')[1]]

def filename_to_category_vector(filename):
    emotion_number = get_emotion_number_from_filename(filename)
    zeros = np.zeros(len(EMOTIONS), dtype='int16')
    zeros[emotion_number] = 1
    return zeros

def get_cached_data(path):
    f = h5py.File(path, 'r')

    kind = path.split('.')[-3]
    print('kind: ', kind)


    names = None
    if kind == 'spectrograms':
        names = spectrogram_names
    elif kind == 'waveforms':
        names = waveform_names

    data = []
    for key in names:
        stuff = f[key]
        if np.issubdtype(stuff.dtype, np.dtype('|S')):
            data.append(np.array([x.decode('UTF-8') for x in stuff[:]]))
        else:
            data.append(stuff[:])

    return data



def cache_data(path, data):

    names = None
    if len(data) == 4:
        names = waveform_names
    else:
        names = spectrogram_names

    f = h5py.File(path, 'w')

    for name, datum in zip(names, data):
        if np.issubdtype(datum.dtype, np.dtype('<U')):
            datum = [x.encode('ascii') for x in datum]
        f.create_dataset(name, data=datum)

    f.close()


def get_cached_ttv_data(path):
    f = h5py.File(path, 'r')

    return (
    {
        'x': f['test']['x'][:],
        'y': f['test']['y'][:]
    },
    {
        'x': f['train']['x'][:],
        'y': f['train']['y'][:],
    },
    {
        'x': f['validation']['x'][:],
        'y': f['validation']['y'][:],
    }
    )


def cache_ttv_data(path, ttv_data):
    test_data, train_data, validation_data = ttv_data

    f = h5py.File(path, 'w')

    f.create_dataset('test/x', data=test_data['x'])
    f.create_dataset('test/y', data=test_data['y'])

    f.create_dataset('train/x', data=train_data['x'])
    f.create_dataset('train/y', data=train_data['y'])

    f.create_dataset('validation/x', data=validation_data['x'])
    f.create_dataset('validation/y', data=validation_data['y'])
    f.close()
