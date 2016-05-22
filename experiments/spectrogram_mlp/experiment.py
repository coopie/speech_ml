import src_context

import learning
from ttv_to_spectrograms import ttv_to_spectrograms
from util import ttv_yaml_to_dict, EMOTIONS

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np
import math
import random

from scipy.interpolate import interp1d

SAMPLE_RATE = 48000
TIME_WINDOW = 1



THIS_DIR = 'experiments/spectrogram_mlp/'

def main():
    ttv_info = ttv_yaml_to_dict(THIS_DIR + 'ttv_brt.yaml')
    print("GETTING SPECTORGRAM DATA...")
    spectrogram_data = ttv_to_spectrograms(
        ttv_info,
        normalise_waveform=normalise,
        normalise_spectrogram=slice_spectrogram,
        cache=THIS_DIR,
    )
    test, train, validation = ttv_data = learning.split_ttv(spectrogram_data)

    learning.train(
        make_mlp_model,
        ttv_data,
        'mlp_spectrogram_model',
        path_to_results=THIS_DIR,
        generate_callbacks=generate_callbacks,
        number_of_epochs=200,
        dry_run=False,
        to_terminal=True
    )

def generate_callbacks():
    return [
        EarlyStopping(monitor='val_acc', patience=30)
    ]

def make_mlp_model(verbosity=1, example_input=np.eye(100), **unused):

    compile_args = {
        'loss': "categorical_crossentropy",
        'optimizer': "RMSprop",
        'metrics': ['accuracy']
    }

    model = Sequential()

    top_layer = random.randint(512, 1024) * 5
    second_layer = random.randint(4, 12) ** 2

    if verbosity >= 1:
        print('top layer: ', top_layer, 'second layer: ', second_layer)

    model.add(Flatten(input_shape=example_input.shape))

    model.add(Dense(top_layer, activation="tanh", init='uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(second_layer, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(len(EMOTIONS), activation="softmax"))

    model.compile(
        **compile_args
    )

    return model, compile_args


def normalise(datum, frequency, **unused):
    return datum[-(frequency*TIME_WINDOW):]


def slice_spectrogram(spec, frequencies):
    return spec[int(len(spec) * 0.75) :]

if __name__ == '__main__':
    main()
