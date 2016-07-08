import unittest
import os
import numpy as np

from speech_ml.metrics import confusion_matrix_metric

DUMMY_DATA_PATH = os.path.join('test', 'dummy_data', 'metadata')


class MetricsTests(unittest.TestCase):
    def test_confusion_matrix(self):

        # simplest example
        targets = np.array([[1], [0]])
        predictions = np.array([[1], [0]])

        [conf_matrix], [conf_matrix_name] = confusion_matrix_metric(targets, predictions)


        expected = np.array([
            [1, 0],
            [0, 1]
        ])

        np.testing.assert_equal(conf_matrix, expected)
        self.assertEqual(conf_matrix_name, 'confusion_matrix')


        dirty_preds = np.array([[0.51], [0.49]])
        [conf_matrix], [conf_matrix_name] = confusion_matrix_metric(targets, dirty_preds)
        np.testing.assert_equal(conf_matrix, expected)


        #  increase the threshold so that both are not positively classified
        [conf_matrix], [conf_matrix_name] = confusion_matrix_metric(targets, dirty_preds, threshold=0.51)


        expected = np.array([
            [1, 0],
            [1, 0]
        ])

        np.testing.assert_equal(conf_matrix, expected)

        # large complex one
        def e(index):
            a = np.zeros(3)
            a[index] = 1
            return a



        targets = np.array([
            e(1),
            e(2),
            e(2),
            e(0),
            e(0),
            e(0),
        ])
        a = np.array
        predictions = np.array([
            a([0.1, 0.7, 0.4]),  # 1
            a([0.01, 0.5, 0.4]),  # 1
            a([0.1, 0.1, 0.4]),  # 2
            a([0.7, 0.1, 0.4]),  # 0
            a([0.7, 0.1, 0.4]),  # 0
            a([0.7, 0.0, 0.9]),  # 2
        ])

        expected = np.array([
            [2, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ])
        [conf_matrix], [conf_matrix_name] = confusion_matrix_metric(targets, predictions)
        np.testing.assert_equal(conf_matrix, expected)


if __name__ == '__main__':
    unittest.main()
