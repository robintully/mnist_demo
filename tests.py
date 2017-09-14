import unittest
import os
import app.keras_worker
import model.data_prep
import model.train_model
import model.h5_wrapper

from flask import current_app
import numpy as np


class RoutesTest(unittest.TestCase):
    """Tests for routes"""

    @classmethod
    def setUpClass(cls):
        cls.app = app.app

    def setUp(self):
        with self.app.app_context():
            self.client = current_app.test_client()

    def test_answer_string_route(self):
        with self.app.test_request_context \
                    ('/index/1'):
            response = self.app.full_dispatch_request()
            self.assertTrue(response.status_code == 200)


class KerasWorkerTest(unittest.TestCase):
    """ Run tests on the Keras worker modules"""

    def test_mnist_by_index(self):
        """Validate keras can make a prediction"""
        result = app.keras_worker.mnist_by_index(2)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[0], np.int64)


class DataPrepTest(unittest.TestCase):
    """Validates Data Prep"""
    def test_data_prep_keys(self):
        """Validate keras can make a prediction"""
        keys = ['X_TRAIN', 'Y_TRAIN', 'X_TEST', 'Y_TEST', 'NUM_CLASSES', 'IMG_ROWS', 'IMG_COLS', 'INPUT_SHAPE']
        for key in keys:
            # import pdb; pdb.set_trace()
            self.assertIn(key, dir(model.data_prep))


class TrainModelTest(unittest.TestCase):
    """Validates Train Model"""
    def test_function_to_train(self):
        self.assertIn('train_model', dir(model.train_model))


class H5WrapperTest(unittest.TestCase):
    """Validates H5 train"""
    def tearDown(self):
        os.remove('test.h5')

    def test_function_to_train(self):
        model.h5_wrapper.save_to_h5('test', np.array([1, 2, 3]))
        self.assertTrue(os.path.isfile('test.h5'))
