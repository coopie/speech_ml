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
import os
from waveform_tools import pad_or_slice

TIME_WINDOW = 3

THIS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'


def main():
    ttv_file = 'ttv_bt.yaml'
    ttv_info = ttv_yaml_to_dict(THIS_DIR + ttv_file)
    print("GETTING SPECTORGRAM DATA...")
    spectrogram_data = ttv_to_spectrograms(
        ttv_info,
        normalise_waveform=normalise_waveform,
        normalise_spectrogram=slice_spectrogram,
        cache=THIS_DIR,
    )
    test, train, validation = ttv_data = learning.split_ttv(spectrogram_data)

    learning.train(
        make_mlp_model,
        ttv_data,
        'mlp_spectrogram_' + ttv_file.split('.')[0],
        path_to_results=THIS_DIR,
        generate_callbacks=generate_callbacks,
        number_of_epochs=200,
        dry_run=False
        # to_terminal=True
    )

def generate_callbacks():
    return [
        EarlyStopping(monitor='val_acc', patience=10)
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


def normalise_waveform(datum, frequency, **unused):
    return pad_or_slice(datum, frequency*TIME_WINDOW)
    # return datum[-(frequency*TIME_WINDOW):]


def slice_spectrogram(spec, frequencies,**unused):
    smaller_than_2khz = sum(frequencies < 2000)
    return spec[:smaller_than_2khz]


if __name__ == '__main__':
    main()
