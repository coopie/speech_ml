# library for experiments. Thanks to Timotej Kapus (Github: kren1) for allowing me to
# take heavy influence from his work on Palpitate

import ttv
import kbhit
import code
from kbhit import KBHit
from scipy.stats import pearsonr
import yaml
import os
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import model_from_yaml
import numpy as np


kb = None

def keypress_to_quit():
    global kb

    if kb is None:
        kb = KBHit()

    while kb.kbhit():
        try:
            if "q" in kb.getch():
                print("quiting due to user pressing q")
                return True
        except UnicodeDecodeError:
            pass


def train(
        generate_model,
        ttv,
        experiment_name,
        path_to_results='',
        end_training=keypress_to_quit,
        early_stopping=None,
        to_terminal=False,
        verbosity=1,
        number_of_epochs=100,
        batch_size=100,
        dry_run=False
    ):
    """
    TODO: write this
    """
    callbacks = []
    model_perf_tracker = None
    if not dry_run:
        model_perf_tracker = CompleteModelCheckpoint(
            path_to_results + '/' + experiment_name + '_weights_{val_acc:.4f}',
            monitor='val_acc',
            save_best_only=True
        )
        callbacks.append(model_perf_tracker)

    if early_stopping is not None:
        callbacks.append(early_stopping)

    def log(message, level):
        if verbosity >= level:
            print message

    model = None
    test_data, train_data, validation_data = ttv

    while not end_training():
        model = generate_model()

        history = model.fit(
            train_data['x'],
            train_data['y'],
            batch_size=batch_size,
            nb_epoch=number_of_epochs,
            verbose=1,
            validation_data=(validation_data['x'],
            validation_data['y']),
            callbacks=callbacks
        )
        log("END OF EPOCHS: BEST VAIDATION ACCURACY: {0:4f}".format(max(history.history['val_acc'])), 1)

    log('TRAINING ENDED, GETTING TEST SET RESULTS', 1)

    best_model = None
    if not dry_run:

        path_to_best = model_perf_tracker.path_to_best
        best_model_config = open(path_to_best + '.yaml').read()

        if best_model_config == model.to_yaml():
            log('BEST MODEL IS SAME CONIFIG AS CURRENT ONE, SAVING TIME, USING SAME MODEL', 1)
            best_model = model
            model.load_weights(path_to_best + '.hdf5')
        else:
            log('BEST MODEL IS NOT SAME CONIFIG AS CURRENT ONE, LOADING NEW MODEL...', 1)
            # del model
            best_model = model_from_yaml(best_model_config)
            best_model.load_weights(path_to_best + '.hdf5')
            log('LOADED', 1)

        experiment_data = {}

        def get_perf(set_and_name):
            data, name = set_and_name
            perf_data = {}
            metrics = model.evaluate(data['x'], data['y'], batch_size=batch_size)
            metrics_names = model.metrics_names

            for name, metric in zip(metrics_names, metrics):
                log((name, name, metric), 1)
                perf_data[name] = float(metric)
            return perf_data

        test_perf, train_perf, validation_perf = map(
            get_perf,
            zip(ttv, ['test', 'train', 'validation'])
        )

        experiment_data['metrics'] = {
            'test': test_perf,
            'train': train_perf,
            'validation': validation_perf
        }

        with open(path_to_results + '/' + experiment_name + '_stats.yaml', 'w') as f:
            yaml.dump(experiment_data, f, default_flow_style=False)

        # save_experiment_results(model, perf_metrics, path_to_results + '/' + experiment_name)
        log('EXPERIEMENT ENDED', 1)

    if to_terminal:
        os.system('reset')
        code.interact(local=locals())


def save_experiment_results(model, results,  path):
    yaml_string = model.to_yaml()
    with open(path + '.yaml', 'w') as f:
        f.write(yaml_string)
        f.close()
    with open(path + '_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    model.save_weights(path + '.h5')


class CompleteModelCheckpoint(Callback):
    '''Save the model after every epoch.

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
    '''
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
                    self.model.save_weights(filepath + '.hdf5', overwrite=True)
                    save_to_file(filepath + '.yaml', self.model.to_yaml())
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            self.model.save_weights(filepath + '.hdf5', overwrite=True)
            save_to_file(filepath + '.yaml', self.model.to_yaml())


def save_to_file(filepath, string):
    with open(filepath, 'w') as f:
        f.write(string)
