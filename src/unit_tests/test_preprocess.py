import configparser
import os
import unittest
import sys

sys.path.insert(1, os.path.join(os.getcwd(), 'src'))

from preprocess import DataMaker

config = configparser.ConfigParser()
config.read('config.ini')


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.data_maker = DataMaker()
        
    def test_preprocessing_data(self):
        self.assertEqual(self.data_maker.preprocessing_data(), True)

if __name__ == '__main__':
    unittest.main()