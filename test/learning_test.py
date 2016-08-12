import unittest
import os

import numpy as np
import h5py

from keras.models import Model
from keras.layers import Dense, Input


from speech_ml.learning import build_model_from_config, get_weights_from_h5_group

TEST_FILE_NAME = 'test_model.hdf5'


class LearningTests(unittest.TestCase):


    def test_build_model_name(self):
        input_img = Input(shape=(1,))

        layer_weights = [np.array([[1337]]), np.array([-666])]
        h = Dense(1, weights=layer_weights, name='cut_me')(input_img)
        output = Dense(2)(h)

        model = Model(input_img, output)

        # COMPILE

        model.save(TEST_FILE_NAME)

        model_h5 = h5py.File(TEST_FILE_NAME)

        rebuilt_h, input_img = build_model_from_config(model_h5, cutoff_layer_name='cut_me')


        import code
        code.interact(local=locals())
        layer_shape = dim_to_tuple(rebuilt_h.get_shape())
        self.assertEqual(layer_shape, (None, 1))

        rebuilt_model = Model(input_img, rebuilt_h)


        # check the right amount of layers were taken
        self.assertEqual(len(rebuilt_model.layers), 2)

        # check the final layer is the one we requested
        self.assertEqual(rebuilt_model.layers[-1].name, 'cut_me')

        # check the weights were loaded
        get_weights_from_h5_group(rebuilt_model, model_h5['model_weights'])
        self.assertEqual(rebuilt_model.layers[-1].get_weights(), layer_weights)



        # check the same works with numbering the layer to slice
        rebuilt_h, input_img = build_model_from_config(model_h5, number_of_layers=1)

        # self.assertEqual(rebuilt_model.name, 'cut_me')

        layer_shape = dim_to_tuple(rebuilt_h.get_shape())
        self.assertEqual(layer_shape, (None, 1))

        rebuilt_model = Model(input_img, rebuilt_h)

        # check the right amount of layers were taken (2 including the input layer)
        self.assertEqual(len(rebuilt_model.layers), 2)



    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_FILE_NAME):
            os.remove(TEST_FILE_NAME)



def dim_to_tuple(ds):
    return tuple([x.value for x in ds])


if __name__ == '__main__':
    unittest.main()
