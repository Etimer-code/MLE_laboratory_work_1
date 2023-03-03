import configparser
import os
import unittest
import sys

sys.path.insert(1, os.path.join(os.getcwd(), 'src'))

from train import CNN_Model

config = configparser.ConfigParser()
config.read('config.ini')


class Test_CNN_Model(unittest.TestCase):

    def setUp(self) -> None:
        self.cnn_model = CNN_Model()
    
    def test_cnn(self):
        self.assertEqual(self.cnn_model.cnn(use_config = False), True)

if __name__ == '__main__':
    unittest.main()