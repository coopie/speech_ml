from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import h5py

from libcontext import util
from util import EMOTIONS, EMOTION_NUMBERS, mkdir_p

DATA_DIR = 'data/RAVDESS'

def make_output_vector(dataset_name):
    info = dataset_name.split('_')
    emotion_number = EMOTION_NUMBERS[info[1]]
    result_vector = np.zeros(len(EMOTIONS))
    result_vector[emotion_number] = 1.0
    return result_vector



# get the data
f = h5py.File(DATA_DIR + "/waveforms.hdf5", "r")

inputs = np.array([np.array(f[dataset]) for dataset in f])
outputs = [make_output_vector(dataset_name) for dataset_name in f]

# For the time being, trim each audio file to be the same length as the smallest
smallest_length = len(inputs[0])
for input in inputs:
    print 'largest thing in the thingy is: ' + str(np.max(input))
    smallest_length = min(smallest_length, len(input))

inputs = [input[0:smallest_length] for input in inputs]

inputs = np.array(inputs).astype('float32')
outputs = np.array(outputs).astype('float32')

model = Sequential()
model.add(Dense(512, activation="sigmoid", input_shape=(smallest_length,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_absolute_error", optimizer="rmsprop")

model.fit(inputs, outputs, nb_epoch=10000)
