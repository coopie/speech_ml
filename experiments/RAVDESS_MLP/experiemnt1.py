import src_context

import learning
from ttv_to_waveforms import ttv_to_waveforms
from util import ttv_yaml_to_dict, EMOTIONS

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
import numpy as np
import math

SAMPLE_RATE = 48000
TIME_WINDOW = 3

THIS_DIR = 'experiments/RAVDESS_MLP/'

def main():
    ttv_info = ttv_yaml_to_dict(THIS_DIR + 'ttv1.yaml')
    print "GETTING WAVEFORM DATA..."
    ttv_data = ttv_to_waveforms(ttv_info, normalise=normalise, cache=THIS_DIR + 'ttv1.cache.hdf5')

    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    learning.train(
        make_mlp_model,
        ttv_data,
        'experiment1',
        path_to_results='experiments/RAVDESS_MLP',
        early_stopping=early_stopping,
        number_of_epochs=1,
        # dry_run=True,
        to_terminal=True
    )


def make_mlp_model():
    model = Sequential()

    model.add(Dense(512, activation="tanh", init='uniform', input_shape=(SAMPLE_RATE*TIME_WINDOW,)))
    model.add(Dropout(0.5))

    model.add(Dense(64, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(len(EMOTIONS), activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=['accuracy']
    )
    return model


def normalise(datum):
    return map(squash, datum[:SAMPLE_RATE*TIME_WINDOW])
# def normalise(datum):
#     return datum[:SAMPLE_RATE*TIME_WINDOW]

def squash(x):
    # no academia backing this up - just a thought
    return math.log(abs(x) + 1, 30)


if __name__ == '__main__':
    main()
