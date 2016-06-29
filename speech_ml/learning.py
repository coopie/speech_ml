# library for experiments. Thanks to Timotej Kapus (Github: kren1) for allowing me to
# take heavy influence from his work on Palpitate

import code
import yaml
import os
from keras.callbacks import Callback
from keras.models import model_from_yaml
from keras.optimizers import get as get_optimizer
from sklearn.metrics import confusion_matrix
import warnings
import numpy as np


from .kbhit import KBHit
from .yaml_util import folded_str
from .util import mkdir_p, save_to_yaml_file, yaml_to_dict
from .data_names import *

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
        test_gen,
        generate_model,
        experiment_name,
        path_to_results='',
        end_training=keypress_to_quit,
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
    # callbacks = generate_callbacks()
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
    example_input = train_gen.data_source[0]

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


def evaluate_model_on_ttv(model, ttv_gens, get_metrics=None, path=False, verbosity=0):

    def log(message, level):
        if verbosity >= level:
            print(message)

    def get_perf(set_and_name):
        # Evaluates a dataset using the metrics that the model uses
        data_gen, name = set_and_name
        perf_data = {}

        metrics = model.evaluate_generator(data_gen, len(data_gen))
        metrics_names = model.metrics_names
        if get_metrics is not None:
            extra_metrics, extra_metrics_names = get_metrics(data_gen, model)
            metrics. += extra_metrics
            metrics_names += extra_metrics_names

        for metric_name, metric in zip(metrics_names, metrics):
            log((name, metric_name, metric), 1)
            if np.size(metric) == 1:
                perf_data[metric_name] = float(metric)
            else:
                perf_data[metric_name] = folded_str(str(metric))
        return perf_data

    test_perf, train_perf, validation_perf = list(map(
        get_perf,
        list(zip(ttv_gens, ['test', 'train', 'validation']))
    ))
    experiment_data = {}
    experiment_data['model'] = path
    experiment_data['metrics'] = {
        'test': test_perf,
        'train': train_perf,
        'validation': validation_perf
    }


    if path is not None:
        with open(path + '/stats.yaml', 'w') as f:
            yaml.dump(experiment_data, f, default_flow_style=False)

    return experiment_data


def save_experiment_results(model, results, path):
    yaml_string = model.to_yaml()
    with open(path + '.yaml', 'w') as f:
        f.write(yaml_string)
        f.close()
    with open(path + '_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    model.save_weights(path + '.h5')


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



def confusion_matrix_metric(model, data_gen):
    """A common metric for classification."""
    y_true = np.nonzero(data['y'])[1]
    y_pred = model.predict_classes(data['x'])
    conf_matrix = confusion_matrix(y_true, y_pred)
    return [conf_matrix], ['confusion_matrix']


def save_to_file(filepath, string):
    with open(filepath, 'w') as f:
        f.write(string)
