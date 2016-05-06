import src_context

import learning
from ttv_to_waveforms import ttv_to_waveforms
from util import ttv_yaml_to_dict, EMOTIONS

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
import numpy as np

SAMPLE_RATE = 48000
TIME_WINDOW = 3

def main():
    ttv_info = ttv_yaml_to_dict('experiments/RAVDESS_MLP/ttv1.yaml')
    print "GETTING WAVEFORM DATA..."
    ttv_data = ttv_to_waveforms(ttv_info, normaliser=normalise)

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
    return datum[:SAMPLE_RATE*TIME_WINDOW]


if __name__ == '__main__':
    main()
