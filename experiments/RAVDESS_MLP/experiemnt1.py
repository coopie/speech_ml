import src_context

import learning
from ttv_to_waveforms import ttv_to_waveforms
from util import ttv_yaml_to_dict

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

SAMPLE_RATE = 48000
TIME_WINDOW = 3

def main():
    ttv_info = ttv_yaml_to_dict('experiments/RAVDESS_MLP/ttv1.yaml')
    ttv_data = ttv_to_waveforms(ttv_info, normaliser=normalise)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    learning.train(
        make_mlp_model,
        ttv_data,
        'experiment1',
        path_to_results='experiments/RAVDESS_MLP',
        early_stopping=early_stopping,
        number_of_epochs=1
    )


def make_mlp_model():
    model = Sequential()
    model.add(Dense(512, activation="sigmoid", input_shape=(SAMPLE_RATE*TIME_WINDOW,)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mean_absolute_error", optimizer="rmsprop")
    return model


def normalise(datum):
    return datum[:SAMPLE_RATE*TIME_WINDOW]


if __name__ == '__main__':
    main()
