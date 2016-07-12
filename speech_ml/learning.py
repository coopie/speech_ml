# library for experiments. Thanks to Timotej Kapus (Github: kren1) for allowing me to
# take heavy influence from his work on Palpitate

import code
import yaml
import os
import warnings
import numpy as np
from collections import Iterable
from tqdm import trange

from keras.callbacks import Callback
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


def one_iteration(iterations, *args, **kwargs):
    return iterations >= 1


def train(
        train_gen,
        validation_gen,
        test_gen,
        generate_model,
        experiment_name,
        path_to_results='',
        end_training=one_iteration,
        early_stopping=None,
        generate_callbacks=empty_list,
        to_terminal=False,
        verbosity=1,
        number_of_epochs=100,
        dry_run=False,
        class_weight=None,
        get_metrics=None
):
    """
    TODO: write this.
    """

    def log(message, level):
        if verbosity >= level:
            print(message)

    if dry_run is True and to_terminal is True:
        warnings.warn('Warning: cannot finish to terminal if you aren\'t saving models')
        to_terminal = False

    model_perf_tracker = None
    if not dry_run:
        model_perf_tracker = CompleteModelCheckpoint(
            os.path.join(path_to_results, experiment_name + '_{val_acc:.4f}_{acc:.4f}'),
            monitor='val_acc',
            save_best_only=True
        )

    manual_stop = ManualEarlyStopping()

    model = None
    example_input = train_gen.data_source_x[0]
    # TODO: example_output

    iterations = 0
    while not end_training(iterations, manual_stop=manual_stop.stopped):
        iterations += 1
        model, compile_args = generate_model(verbosity=verbosity, example_input=example_input, iterations=iterations)

        callbacks = generate_callbacks()
        if not dry_run:
            callbacks.append(model_perf_tracker)
        callbacks.append(manual_stop)

        history = model.fit_generator(
            train_gen,
            samples_per_epoch=len(train_gen),
            validation_data=validation_gen,
            nb_val_samples=len(validation_gen),
            nb_epoch=number_of_epochs,
            verbose=verbosity,
            callbacks=callbacks,
            class_weight=class_weight
        )
        log("END OF EPOCHS: BEST VAIDATION ACCURACY: {0:4f}".format(max(history.history['val_acc'])), 1)
        del model

    log('TRAINING ENDED, GETTING TEST SET RESULTS', 1)

    best_model = None
    if not dry_run:

        path_to_best = model_perf_tracker.path_to_best
        log('LOADING BEST MODEL', 1)
        best_model = load_model(path_to_best)
        log('LOADED', 1)

        experiment_data = evaluate_model_on_ttv(
            best_model,
            (test_gen, train_gen, validation_gen),
            path=path_to_best,
            verbosity=1,
            get_metrics=get_metrics
        )
        log('EXPERIEMENT ENDED', 1)

    if to_terminal:
        os.system('reset')
        code.interact(local=locals())


def load_model(path_to_model_dir):
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


def build_model_from_config(config, weights, number_of_layers, verbosity=1):
    """Build a model from a model snapshot up to `number_of_layers` layers."""
    def log(level, *message):
        if verbosity >= level:
            print(message)

    layers_config = config['config']['layers']

    log(1, 'getting first layer')
    input_layer = layer_from_config(layers_config[0])
    input_img = Input(input_layer.input_shape[1:])
    model = input_img

    log(1, 'after init:', model.get_shape())

    for i, layer_config in enumerate(layers_config[1:]):
        if i >= number_of_layers:
            break
        else:
            # load the weights from the hdf5 file
            class_name = layer_config['class_name']
            if class_name.startswith('Convolution'):
                layer_config['config']['weights'] = [weights[layer_config['name']][w][:] for w in weights[layer_config['name']]]

            model = layer_from_config(layer_config)(model)
            log(1, 'shape:', model.get_shape())

    return model, input_img


def save_model(path, model):
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


class CompleteModelCheckpoint(Callback):
    """Save the model after every epoch.

    This is a slightly adapted version from the keras library. It also saves the
    yaml of the model config.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the validation loss will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minization of the monitored. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto'):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.path_to_best = None


        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.path_to_best = filepath
                    save_model(filepath, self.model)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            save_model(filepath, self.model)


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
