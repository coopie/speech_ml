# library for experiments. Thanks to Timotej Kapus (Github: kren1) for allowing me to
# take heavy influence from his work on Palpitate

import ttv
import kbhit
import code
from kbhit import KBHit
from scipy.stats import pearsonr
import yaml


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
        verbosity='normal',
        number_of_epochs=100,
        batch_size=100
    ):
    """
    TODO: write this
    verbosity: 'normal', 'quiet', 'silent'
    """
    callbacks = [early_stopping] if early_stopping is not None else []

    best_model = None
    test_data, train_data, validation_data = ttv

    while not end_training():
        model = generate_model()

        model.fit(train_data['x'], train_data['y'], batch_size=batch_size, nb_epoch=number_of_epochs,
                verbose=1, validation_data=(validation_data['x'], validation_data['y']), callbacks=callbacks)

        # perf_metrics = assess_model(model, validation_data['x'], validation_data['y'])


        if verbosity in ['normal']:
            # print(str(perf_metrics))
            print("===========================================================")

        # if perf_metrics.better_than(best_perf_metrics):
        #     best_perf_metrics = perf_metrics

    # test_perf_metrics = assess_model(model, test_data.x, test_data.y)
    perf_metrics = 'performance metrics go here'.split(' ')

    save_experiment_results(model, perf_metrics, path_to_results + '/' + experiment_name)


    if to_terminal:
        code.interact(local=locals())


def assess_model(model, X_test, Y_test):
    pass
    # predictions = model.predict(X_test)
    # r = pearsonr(predictions[:,0], Y_test[:,0])
    # rmse = sqrt(mean_squared_error(predictions, Y_test))
    # return r, rmse, predictions


def mean_squared_error(a, b):
    return ((A - B) ** 2).mean(axis=None)


def save_experiment_results(model, results,  path):
    yaml_string = model.to_yaml()
    with open(path + '.yaml', 'w') as f:
        f.write(yaml_string)
        f.close()
    with open(path + '_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    model.save_weights(path + '.h5')
