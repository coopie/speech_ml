from sklearn.metrics import confusion_matrix
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
