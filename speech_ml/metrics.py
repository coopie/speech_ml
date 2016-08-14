from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np


def confusion_matrix_metric(targets, predictions, threshold=0.5):
    """
    Compute confusion matrix.

    Works for arbitrary number of classes. If the shape of the data is one,
    treat as a binary classification with `threshold` as the cutoff point.
    """
    assert targets.ndim == predictions.ndim == 2
    assert targets.shape == predictions.shape

    if targets.shape[1] == 1:
        targets = targets > threshold
        predictions = predictions > threshold
    else:
        targets = np.argmax(targets, axis=1)
        predictions = np.argmax(predictions, axis=1)

    targets = targets.flatten()
    predictions = predictions.flatten()

    conf_matrix = confusion_matrix(targets, predictions)
    return [conf_matrix], ['confusion_matrix']


def roc_curve_metric(targets, predictions, **multi_optional_args):
    """
    For multi class classifiers, two different types of roc curves are made.

    TODO: explain what these are
    """
    assert targets.ndim == predictions.ndim == 2
    assert targets.shape == predictions.shape

    if targets.shape[1] == 1:
        predictions = predictions.flatten()
        targets = targets.flatten()

        roc_for_data = roc_curve(targets, predictions)
    else:
        roc_for_data = {}
        targets = np.argmax(targets, axis=1).flatten()
        for i in range(predictions.shape[1]):
            roc_for_data[i] = {}
            predictions_for_class = predictions[:, i]
            roc_for_data[i]['specific_classification'] = roc_curve(targets, predictions_for_class, pos_label=i)

            predictions_for_class = np.zeros(len(predictions))
            highest_class = np.argmax(predictions, axis=1)
            predictions_for_class[highest_class == i] = predictions[:, highest_class]
            roc_for_data[i]['general_classification'] = roc_curve(targets, predictions_for_class, pos_label=i)

    return [roc_for_data], ['roc_curve']
