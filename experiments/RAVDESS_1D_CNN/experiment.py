#
#   This experiment yielded nothing, trained to learn random chance
#

import src_context

import learning
from ttv_to_waveforms import ttv_to_waveforms
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

THIS_DIR = 'experiments/RAVDESS_1D_CNN/'

def main():
    ttv_info = ttv_yaml_to_dict(THIS_DIR + 'ttv1.yaml')
    print("GETTING WAVEFORM DATA...")
    ttv_data = ttv_to_waveforms(ttv_info, normalise=normalise, cache=THIS_DIR + 'ttv1.cache.hdf5')

    test, train, val = ttv_data
    test['x'] = np.reshape(test['x'], test['x'].shape + (1,))
    train['x'] = np.reshape(train['x'], train['x'].shape + (1,))
    val['x'] = np.reshape(val['x'], val['x'].shape + (1,))


    learning.train(
        make_mlp_model,
        ttv_data,
        '1D_CNN_RAVDESS',
        path_to_results='experiments/RAVDESS_MLP',
        generate_callbacks=generate_callbacks,
        number_of_epochs=200,
        dry_run=True,
        to_terminal=True
    )

def generate_callbacks():
    return [
        EarlyStopping(monitor='val_acc', patience=3)
    ]

def make_mlp_model(**kwargs):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    compile_args = {
        'loss': "categorical_crossentropy",
        'optimizer': sgd,
        'metrics': ['accuracy']
    }

    model = Sequential()

    # if kwargs['verbosity'] >= 1:
        # print('top layer: ', top_layer, 'second layer: ', second_layer)

    model.add(Convolution1D(1, 1000, input_shape=(SAMPLE_RATE*TIME_WINDOW,1)))
    model.add(Activation('relu'))

    model.add(Convolution1D(1, 100))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution1D(1, 10))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))

    model.add(Dense(len(EMOTIONS)))
    model.add(Activation('softmax'))

    model.compile(
        **compile_args
    )
    return model, compile_args


def normalise(datum):
    return list(map(squash, datum[:SAMPLE_RATE*TIME_WINDOW]))

def squash(x):
    # no academia backing this up - just a thought
    return math.log(abs(x) + 1, 30)


if __name__ == '__main__':
    main()
