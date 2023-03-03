import argparse
import configparser
from datetime import datetime
import os
import shutil
import sys
import time
import traceback
import yaml
import tensorflow as tf

from logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read('config.ini')
        
        self.parser = argparse.ArgumentParser(description = 'Predictor')
        self.parser.add_argument('-t',
                                 '--tests',
                                 type = str,
                                 help = 'Select tests',
                                 required = True,
                                 default = 'smoke',
                                 const = 'smoke',
                                 nargs = '?',
                                 choices = ['smoke', 'func'])
        
        self.train_data_set = tf.keras.preprocessing.image_dataset_from_directory(
            directory = self.config['PREPROCESSED_DATA']['train_data'],
            label_mode = self.config['CNN']['LABEL_MODE'],
            batch_size = self.config.getint('CNN', 'BATCH_SIZE'),
            image_size = (self.config.getint('CNN', 'IMAGE_SIZE_1d'), self.config.getint('CNN', 'IMAGE_SIZE_2d')),
            validation_split = self.config.getfloat('CNN', 'TEST_SIZE'),
            subset = 'training',
            seed = self.config.getint('CNN', 'SEED')
        )

        self.validation_data_set = tf.keras.preprocessing.image_dataset_from_directory(
            directory = self.config['PREPROCESSED_DATA']['train_data'],
            label_mode = self.config['CNN']['LABEL_MODE'],
            batch_size = self.config.getint('CNN', 'BATCH_SIZE'),
            image_size = (self.config.getint('CNN', 'IMAGE_SIZE_1d'), self.config.getint('CNN', 'IMAGE_SIZE_2d')),
            validation_split = self.config.getfloat('CNN', 'TEST_SIZE'),
            subset = 'validation',
            seed = self.config.getint('CNN', 'SEED')
        )
        
        self.log.info('Predictor is ready')

    def predict(self) -> bool:
        
        args = self.parser.parse_args()
        
        try:
            classifier = tf.keras.models.load_model(self.config['CNN']['path'])
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        if args.tests == 'smoke':
            try:
                results = classifier.evaluate(self.validation_data_set)
                print(f'CNN has {results[0]} loss and {results[1]} accuracy score')
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(f"{self.config['CNN']['path']} passed smoke tests")
        elif args.tests == 'func':
            tests_path = os.path.join(os.getcwd(), 'tests')
            exp_path = os.path.join(os.getcwd(), 'experiments')
            
            for test in os.listdir(tests_path):
                try:
                    test_dir = os.path.join(tests_path, test)
                    test_data_set = tf.keras.preprocessing.image_dataset_from_directory(
                        directory = test_dir,
                        label_mode = 'categorical',
                        batch_size = 5,
                        image_size = (128, 128)
                    )
                    results = classifier.evaluate(test_data_set)
                    print(f'CNN has {results[0]} loss and {results[1]} accuracy score')
                except Exception:
                    self.log.error(traceback.format_exc())
                    sys.exit(1)
                self.log.info(f"{self.config['CNN']['path']} passed func test {test}")
                exp_data = {'model' : 'CNN',
                            'model params' : dict(self.config.items('CNN')),
                            'tests' : args.tests,
                            'accuracy score' : str(results[1]),
                            'test dir name' : test,
                            'test dir path' : test_dir,
                           }
                date_time = datetime.fromtimestamp(time.time())
                str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                exp_dir = os.path.join(exp_path, f'exp_{test}_{str_date_time}')
                os.mkdir(exp_dir)
                with open(os.path.join(exp_dir, 'exp_config.yaml'), 'w') as exp_f:
                    yaml.safe_dump(exp_data, exp_f, sort_keys = False)
                    shutil.copy(os.path.join(os.getcwd(), 'logfile.log'), 
                                os.path.join(exp_dir, 'exp_logfile.log'))
                    shutil.copytree(self.config['CNN']['path'], os.path.join(exp_dir, f'exp_CNN_sav'))
        return True



if __name__ == '__main__':
    predictor = Predictor()
    predictor.predict()