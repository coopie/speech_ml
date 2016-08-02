"""Library for experiments.
"""
import yaml
import os
import warnings
import numpy as np
from collections import Iterable
from tqdm import trange

from keras.callbacks import Callback, ModelCheckpoint
from keras.models import model_from_yaml
from keras.optimizers import get as get_optimizer
from keras.utils.layer_utils import layer_from_config
from keras.layers import Input


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
        experiment_name,
        path_to_results='',
        early_stopping=None,
        generate_callbacks=empty_list,
        verbosity=1,
        number_of_epochs=100,
        dry_run=False,
        class_weight=None,
):
    """
    TODO: write this.
    """

    def log(message, level):
        if verbosity >= level:
            print(message)

    if not dry_run:
        model_perf_tracker = ModelCheckpoint(
            os.path.join(path_to_results, experiment_name + '_{val_acc:.4f}_{acc:.4f}.hdf5'),
            monitor='val_loss',
            save_best_only=True
        )

    manual_stop = ManualEarlyStopping()

    example_input = train_gen.data_source_x[0]
    # TODO: example_output
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


def build_model_from_config(config, weights, cutoff_layer_name=None, number_of_layers=None, verbosity=1):
    """Build a model from a model snapshot up to `number_of_layers` layers, or `cutoff_layer_name` is reached.

    Note that this currently only works for convolutional architectures
    """
    def log(level, *message):
        if verbosity >= level:
            print(message)

    if cutoff_layer_name is None and number_of_layers is None:
        raise RuntimeError('You must either name a cutoff layer or the number of layers to cut to')

    layers_config = config['config']['layers']

    log(1, 'getting first layer')
    input_layer = layer_from_config(layers_config[0])
    input_img = Input(input_layer.input_shape[1:])
    model = input_img

    log(1, 'after init:', model.get_shape())

    if number_of_layers is None:
        number_of_layers = float('inf')

    for i, layer_config in enumerate(layers_config[1:]):
        if i >= number_of_layers:
            break

        # load the weights from the hdf5 file
        class_name = layer_config['class_name']
        if class_name.startswith('Convolution'):
            layer_config['config']['weights'] = [weights[layer_config['name']][w][:] for w in weights[layer_config['name']]]

        model = layer_from_config(layer_config)(model)
        if layer_config['name'] == cutoff_layer_name:
            break
        log(1, 'shape:', model.get_shape())


    return model, input_img


def save_model(path, model):
    warnings.warn('`save_model` called. This is a deprected function!')
    metrics = model.metrics_names.copy()
    metrics.remove('loss')
    compile_args = {
        'loss': model.loss,
        'metrics': metrics,
        'optimizer': model.optimizer.get_config()
    }
    mkdir_p(path)
    model.save_weights(path + '/weights.hdf5', overwrite=True)
    save_to_file(path + '/config.yaml', model.to_yaml())
    save_to_yaml_file(path + '/compile_args.yaml', compile_args)


def evaluate_model_on_ttv(model, ttv_gens, sets=['test', 'train', 'validation'], get_metrics=None, path=False, verbosity=0):
    warnings.warn('`evaluate_model_on_ttv` called. This is a deprected function!')
    assert len(ttv_gens) == len(sets)

    def log(message, level):
        if verbosity >= level:
            print(message)

    def get_perf(set_and_name):
        # Evaluates a dataset using the metrics that the model uses
        data_gen, name = set_and_name
        log('Evaluating ' + name, 1)
        perf_data = {}

        metrics = model.evaluate_generator(data_gen, len(data_gen))
        metrics_names = model.metrics_names
        if get_metrics is not None:
            log('Getting custom metrics', 1)
            y_true = None
            y_pred = None
            samples_seen = 0

            for i in trange(len(data_gen) // data_gen.batch_size):
                Xs, Ys = next(data_gen)
                samples_seen += len(Ys)
                predictions = model.predict_on_batch(Xs)

                if y_true is None and y_pred is None:
                    y_true = Ys
                    y_pred = predictions
                else:
                    y_true = np.append(y_true, Ys, axis=0)
                    y_pred = np.append(y_pred, predictions, axis=0)


            extra_metrics, extra_metrics_names = get_metrics(y_true, y_pred)
            metrics += extra_metrics
            metrics_names += extra_metrics_names

        for metric_name, metric in zip(metrics_names, metrics):
            def log_metric(m, m_name):
                if isinstance(m, dict):
                    log(str(m_name) + ': ', 1)
                    for key, value in m.items():
                        log_metric(value, key)
                elif isinstance(m, tuple):
                    log_metric(list(m), m_name)
                elif np.size(m) <= 64:  # don't log massive metrics (e.g. roc curves)
                    log((m_name, m), 1)
                else:
                    log((m_name, 'TOO BIG TO PRINT'), 1)
            log_metric(metric, metric_name)
            perf_data[metric_name] = format_for_hr_yaml(metric)

        return perf_data

    perfs = list(map(
        get_perf,
        list(zip(ttv_gens, sets))
    ))
    experiment_data = {}
    experiment_data['model'] = path
    experiment_data['metrics'] = dict(zip(sets, perfs))

    if path is not None:
        with open(path + '/stats.yaml', 'w') as f:
            yaml.dump(experiment_data, f, default_flow_style=False)

    return experiment_data


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


def format_for_hr_yaml(metric):
    """Used so that the metrics from model evaluation are in a human readable format."""
    if isinstance(metric, np.ndarray) and np.size(metric) == 1:
        return float(metric)
    elif isinstance(metric, dict):
        return {label: format_for_hr_yaml(x) for label, x in metric.items()}
    elif isinstance(metric, np.ndarray):
        return folded_str(str(metric))
    elif isinstance(metric, Iterable):
        return [format_for_hr_yaml(x) for x in metric]
    else:
        return folded_str(str(metric))
