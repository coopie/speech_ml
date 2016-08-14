"""Library for experiments.
"""
import yaml
import os
import warnings
import numpy as np
from collections import Iterable
from tqdm import trange
import json

from keras.callbacks import Callback, ModelCheckpoint
from keras.models import model_from_config
from keras.optimizers import get as get_optimizer
from keras.utils.layer_utils import layer_from_config
from keras.layers import Input
from keras import backend as K


from .kbhit import KBHit
from .yaml_util import folded_str
from .util import mkdir_p, save_to_yaml_file, yaml_to_dict

kb = None


def keypress_to_quit(key='q', manual_stop=False, *unused):
    global kb
    if manual_stop:
        return True

    if kb is None:
        kb = KBHit()

    while kb.kbhit():
        try:
            if key in kb.getch():
                return True
        except UnicodeDecodeError:
            pass


def empty_list():
    return []


def train(
        train_gen,
        validation_gen,
        generate_model,
        experiment_name=None,
        path_to_results='',
        early_stopping=None,
        generate_callbacks=empty_list,
        verbosity=1,
        number_of_epochs=100,
        dry_run=False,
        class_weight=None,
):
    """
    A thin wrapper around `model.fit_generator` with some defaults that reduce bloat in .
    """

    def log(message, level):
        if verbosity >= level:
            print(message)

    if not dry_run:
        assert experiment_name is not None, 'When saving models during training, you must set `experiment_name`.'

        model_perf_tracker = ModelCheckpoint(
            os.path.join(path_to_results, experiment_name + '_{val_acc:.4f}_{acc:.4f}.hdf5'),
            monitor='val_loss',
            save_best_only=True
        )

    manual_stop = ManualEarlyStopping()

    example_input = train_gen.data_source_x[0]
    batch_size = train_gen.batch_size


    model, compile_args = generate_model(
        verbosity=verbosity,
        example_input=example_input,
        batch_size=batch_size
    )

    callbacks = generate_callbacks()
    if not dry_run:
        callbacks.append(model_perf_tracker)
    callbacks.append(manual_stop)

    model.fit_generator(
        train_gen,
        samples_per_epoch=len(train_gen),
        validation_data=validation_gen,
        nb_val_samples=len(validation_gen),
        nb_epoch=number_of_epochs,
        verbose=verbosity,
        callbacks=callbacks,
        class_weight=class_weight
    )

    return model


def load_model_old(path_to_model_dir):
    warnings.warn('`load_model` called. This is a deprected function!')
    model = model_from_yaml(open(path_to_model_dir + '/config.yaml').read())
    model.load_weights(path_to_model_dir + '/weights.hdf5')
    compile_args = yaml_to_dict(path_to_model_dir + '/compile_args.yaml')

    optimizer = compile_args.pop('optimizer')
    if isinstance(optimizer, dict):
        name = optimizer.pop('name')
        optimizer = get_optimizer(name, optimizer)
    else:
        optimizer = get_optimizer(optimizer)

    model.compile(optimizer=optimizer, **compile_args)
    return model


def build_model_from_config(model_h5, cutoff_layer_name=None, number_of_layers=None, verbosity=1):
    """Build a model from a model snapshot up to `number_of_layers` layers, or `cutoff_layer_name` is reached.

    TODO: option to pass in input size to the rebuilding of the network.
    """
    def log(level, *message):
        if verbosity >= level:
            print(message)

    if cutoff_layer_name is None and number_of_layers is None:
        raise RuntimeError('You must either name a cutoff layer or the number of layers to cut to')

    config_json_str = model_h5.attrs.get('model_config').decode('UTF-8')
    config = json.loads(config_json_str)

    layers_config = config['config']['layers']

    log(1, 'getting first layer')
    input_layer = layer_from_config(layers_config[0])
    input_img = Input(input_layer.input_shape[1:])
    model = input_img

    if number_of_layers is None:
        number_of_layers = float('inf')

    # Build the model until we reach the stopping point
    for i, layer_config in enumerate(layers_config[1:]):
        if i >= number_of_layers:
            break
        model = layer_from_config(layer_config)(model)
        if layer_config['name'] == cutoff_layer_name:
            break
        log(1, 'shape:', model.get_shape())


    return model, input_img


def get_weights_from_h5_group(model, model_weights, verbose=1):
    layers = model.layers

    weight_value_tuples = []
    for layer in layers:
        name = layer.name
        if name in model_weights and len(model_weights[name]) > 0:
            layer_weights = model_weights[name]
            weight_names = [n.decode('utf8') for n in layer_weights.attrs['weight_names']]
            weight_values = [layer_weights[weight_name] for weight_name in weight_names]
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')

            if verbose:
                print('Setting_weights for layer:', name)

            weight_value_tuples += zip(symbolic_weights, weight_values)

    K.batch_set_value(weight_value_tuples)


class ManualEarlyStopping(Callback):
    """
    "e" : exit the training completely
    "r" : stop training this model, and start the next iteration
    """
    def __init__(self):
        super(Callback, self).__init__()
        self.stopped = False

    def on_epoch_end(self, epoch, logs={}):
        global kb
        if kb is None:
            kb = KBHit()

        while kb.kbhit():
            try:
                c = kb.getch()
                if 'e' in c:
                    self.model.stop_training = True
                    self.stopped = True
                elif 'r' in c:
                    self.model.stop_training = True
            except UnicodeDecodeError:
                pass


def save_to_file(filepath, string):
    with open(filepath, 'w') as f:
        f.write(string)
