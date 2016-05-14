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

SAMPLE_RATE = 48000
TIME_WINDOW = 3



THIS_DIR = 'experiments/RAVDESS_SPEC_MLP/'

def main():
    ttv_info = ttv_yaml_to_dict(THIS_DIR + 'ttv.yaml')
    print("GETTING SPECTORGRAM DATA...")
    ttv_data = ttv_to_spectrograms(
        ttv_info,
        normalise_waveform=normalise,
        normalise_spectrogram=slice_spectrogram,
        cache=THIS_DIR + 'ttv'
    )

    test, train, val = ttv_data

    learning.train(
        make_mlp_model,
        ttv_data,
        '1D_CNN_RAVDESS',
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

def make_mlp_model(**kwargs):

    compile_args = {
        'loss': "categorical_crossentropy",
        'optimizer': "RMSprop",
        'metrics': ['accuracy']
    }

    model = Sequential()

    top_layer = random.randint(512, 1024) * 5
    second_layer = random.randint(4, 12) ** 2

    if kwargs['verbosity'] >= 1:
        print('top layer: ', top_layer, 'second layer: ', second_layer)

    model.add(Flatten(input_shape=kwargs['example_input'].shape))

    model.add(Dense(top_layer, activation="tanh", init='uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(second_layer, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(len(EMOTIONS), activation="softmax"))

    model.compile(
        **compile_args
    )
    return model, compile_args


def normalise(datum):
    return datum[:SAMPLE_RATE*TIME_WINDOW]

def slice_spectrogram(spec):
    return spec[len(spec)//2 :]

if __name__ == '__main__':
    main()
