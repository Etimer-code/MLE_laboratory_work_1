import configparser
import os
import pickle
import sys
import traceback

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPool2D, Flatten, Dense

from logger import Logger

SHOW_LOG = True



class CNN_Model():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read('config.ini')
        
        self.experiments_path = os.path.join(os.getcwd(), 'experiments')
        self.cnn_model_path = os.path.join(self.experiments_path, 'cnn_sav')
        self.log.info('CNN_Model is ready')
        
    def cnn(self, use_config : bool, predict = False) -> bool:
        
        params = self.get_CNN_params(use_config)
        try:
            self.train_data_set = tf.keras.preprocessing.image_dataset_from_directory(
                directory = self.config['PREPROCESSED_DATA']['train_data'],
                label_mode = params['LABEL_MODE'],
                batch_size = params['BATCH_SIZE'],
                image_size = (params['IMAGE_SIZE_1d'], params['IMAGE_SIZE_2d']),
                validation_split = params['TEST_SIZE'],
                subset = 'training',
                seed = params['SEED']
            )

            self.validation_data_set = tf.keras.preprocessing.image_dataset_from_directory(
                directory = self.config['PREPROCESSED_DATA']['train_data'],
                label_mode = params['LABEL_MODE'],
                batch_size = params['BATCH_SIZE'],
                image_size = (params['IMAGE_SIZE_1d'], params['IMAGE_SIZE_2d']),
                validation_split = params['TEST_SIZE'],
                subset = 'validation',
                seed = params['SEED']
            )
            classifier = Sequential([
                Input(shape = (params['input_shape_1d'],
                               params['input_shape_2d'], 
                               params['input_shape_3d'])),
                Conv2D(filters = params['1_layer_filters'], 
                       kernel_size = (params['1_layer_kernel_size_1d'], 
                                      params['1_layer_kernel_size_2d']), 
                       activation = params['1_layer_activation']),
                MaxPool2D(pool_size = (params['2_layer_pool_size_1d'], 
                                       params['2_layer_pool_size_2d'])),
                Conv2D(filters = params['3_layer_filters'], 
                       kernel_size = (params['3_layer_kernel_size_1d'], 
                                      params['3_layer_kernel_size_2d']), 
                       activation = params['3_layer_activation']),
                MaxPool2D(pool_size = (params['4_layer_pool_size_1d'], 
                                       params['4_layer_pool_size_2d'])),
                Conv2D(filters = params['5_layer_filters'], 
                       kernel_size = (params['5_layer_kernel_size_1d'], 
                                      params['5_layer_kernel_size_2d']), 
                       activation = params['5_layer_activation']),
                MaxPool2D(pool_size = (params['6_layer_pool_size_1d'], 
                                       params['6_layer_pool_size_2d'])),
                Conv2D(filters = params['7_layer_filters'], 
                       kernel_size = (params['7_layer_kernel_size_1d'], 
                                      params['7_layer_kernel_size_2d']), 
                       activation = params['7_layer_activation']),
                MaxPool2D(pool_size = (params['8_layer_pool_size_1d'],
                                       params['8_layer_pool_size_2d'])),
                Flatten(),
                Dense(params['9_layer_units'],
                      activation = params['9_layer_activation']),
                Dense(params['10_layer_units'],
                      activation = params['10_layer_activation'])
            ])
            
            classifier.compile(optimizer = params['OPTIMIZER'],
                               loss = params['LOSS'],
                               metrics = [params['METRICS']],)
            
            history = classifier.fit(self.train_data_set,
                                     steps_per_epoch = params['STEPS_PER_EPROCH'],
                                     epochs = params['EPOCHS'],
                                     validation_data = self.validation_data_set, 
                                     validation_steps = params['VALIDATION_STEPS'])
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        if predict:
            results = classifier.evaluate(self.validation_data_set)
            print(f'CNN has {results[0]} loss and {results[1]} accuracy score')
        
        params['path'] = self.cnn_model_path
        
        return self.save_model(classifier, self.cnn_model_path, 'CNN', params)
    
    def get_CNN_params(self, use_config : bool) -> dict:
        CNN_params = {}
        if use_config:
            try:
                CNN_params['LABEL_MODE'] = self.config['CNN']['LABEL_MODE']
                CNN_params['BATCH_SIZE'] = self.config.getint('CNN', 'BATCH_SIZE')
                CNN_params['IMAGE_SIZE_1d'] = self.config.getint('CNN', 'IMAGE_SIZE_1d')
                CNN_params['IMAGE_SIZE_2d'] = self.config.getint('CNN', 'IMAGE_SIZE_2d')
                CNN_params['TEST_SIZE'] = self.config.getfloat('CNN', 'TEST_SIZE')
                CNN_params['SEED'] = self.config.getint('CNN', 'SEED')
                CNN_params['input_shape_1d'] = self.config.getint('CNN', 'input_shape_1d')
                CNN_params['input_shape_2d'] = self.config.getint('CNN', 'input_shape_2d')
                CNN_params['input_shape_3d'] = self.config.getint('CNN', 'input_shape_3d')
                CNN_params['1_layer_filters'] = self.config.getint('CNN', '1_layer_filters')
                CNN_params['1_layer_kernel_size_1d'] = self.config.getint('CNN', '1_layer_kernel_size_1d')
                CNN_params['1_layer_kernel_size_2d'] = self.config.getint('CNN', '1_layer_kernel_size_2d')
                CNN_params['1_layer_activation'] = self.config['CNN']['1_layer_activation']
                CNN_params['2_layer_pool_size_1d'] = self.config.getint('CNN', '2_layer_pool_size_1d')
                CNN_params['2_layer_pool_size_2d'] = self.config.getint('CNN', '2_layer_pool_size_2d')
                CNN_params['3_layer_filters'] = self.config.getint('CNN', '3_layer_filters')
                CNN_params['3_layer_kernel_size_1d'] = self.config.getint('CNN', '3_layer_kernel_size_1d')
                CNN_params['3_layer_kernel_size_2d'] = self.config.getint('CNN', '3_layer_kernel_size_2d')
                CNN_params['3_layer_activation'] = self.config['CNN']['3_layer_activation']
                CNN_params['4_layer_pool_size_1d'] = self.config.getint('CNN', '4_layer_pool_size_1d')
                CNN_params['4_layer_pool_size_2d'] = self.config.getint('CNN', '4_layer_pool_size_2d')
                CNN_params['5_layer_filters'] = self.config.getint('CNN', '5_layer_filters')
                CNN_params['5_layer_kernel_size_1d'] = self.config.getint('CNN', '5_layer_kernel_size_1d')
                CNN_params['5_layer_kernel_size_2d'] = self.config.getint('CNN', '5_layer_kernel_size_2d')
                CNN_params['5_layer_activation'] = self.config['CNN']['5_layer_activation']
                CNN_params['6_layer_pool_size_1d'] = self.config.getint('CNN', '6_layer_pool_size_1d')
                CNN_params['6_layer_pool_size_2d'] = self.config.getint('CNN', '6_layer_pool_size_2d')
                CNN_params['7_layer_filters'] = self.config.getint('CNN', '7_layer_filters')
                CNN_params['7_layer_kernel_size_1d'] = self.config.getint('CNN', '7_layer_kernel_size_1d')
                CNN_params['7_layer_kernel_size_2d'] = self.config.getint('CNN', '7_layer_kernel_size_2d')
                CNN_params['7_layer_activation'] = self.config['CNN']['7_layer_activation']
                CNN_params['8_layer_pool_size_1d'] = self.config.getint('CNN', '8_layer_pool_size_1d')
                CNN_params['8_layer_pool_size_2d'] = self.config.getint('CNN', '8_layer_pool_size_2d')
                CNN_params['9_layer_units'] = self.config.getint('CNN', '9_layer_units')
                CNN_params['9_layer_activation'] = self.config['CNN']['9_layer_activation']
                CNN_params['10_layer_units'] = self.config.getint('CNN', '10_layer_units')
                CNN_params['10_layer_activation'] = self.config['CNN']['10_layer_activation']
                CNN_params['OPTIMIZER'] = self.config['CNN']['OPTIMIZER']
                CNN_params['LOSS'] = self.config['CNN']['LOSS']
                CNN_params['METRICS'] = self.config['CNN']['METRICS']
                CNN_params['STEPS_PER_EPROCH'] = self.config.getint('CNN', 'STEPS_PER_EPROCH')
                CNN_params['EPOCHS'] = self.config.getint('CNN', 'EPOCHS')
                CNN_params['VALIDATION_STEPS'] = self.config.getint('CNN', 'VALIDATION_STEPS')
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config : {use_config}, no params')
                sys.exit(1)
        else:
            CNN_params['LABEL_MODE'] = 'categorical'
            CNN_params['BATCH_SIZE'] = 32
            CNN_params['IMAGE_SIZE_1d'] = 128
            CNN_params['IMAGE_SIZE_2d'] = 128
            CNN_params['TEST_SIZE'] = 0.2
            CNN_params['SEED'] = 0
            CNN_params['input_shape_1d'] = 128
            CNN_params['input_shape_2d'] = 128
            CNN_params['input_shape_3d'] = 3
            CNN_params['1_layer_filters'] = 32
            CNN_params['1_layer_kernel_size_1d'] = 3
            CNN_params['1_layer_kernel_size_2d'] = 3
            CNN_params['1_layer_activation'] = 'relu'
            CNN_params['2_layer_pool_size_1d'] = 2
            CNN_params['2_layer_pool_size_2d'] = 2
            CNN_params['3_layer_filters'] = 64
            CNN_params['3_layer_kernel_size_1d'] = 3
            CNN_params['3_layer_kernel_size_2d'] = 3
            CNN_params['3_layer_activation'] = 'relu'
            CNN_params['4_layer_pool_size_1d'] = 2
            CNN_params['4_layer_pool_size_2d'] = 2
            CNN_params['5_layer_filters'] = 128
            CNN_params['5_layer_kernel_size_1d'] = 3
            CNN_params['5_layer_kernel_size_2d'] = 3
            CNN_params['5_layer_activation'] = 'relu'
            CNN_params['6_layer_pool_size_1d'] = 2
            CNN_params['6_layer_pool_size_2d'] = 2
            CNN_params['7_layer_filters'] = 128
            CNN_params['7_layer_kernel_size_1d'] = 3
            CNN_params['7_layer_kernel_size_2d'] = 3
            CNN_params['7_layer_activation'] = 'relu'
            CNN_params['8_layer_pool_size_1d'] = 2
            CNN_params['8_layer_pool_size_2d'] = 2
            CNN_params['9_layer_units'] = 512
            CNN_params['9_layer_activation'] = 'relu'
            CNN_params['10_layer_units'] = 2
            CNN_params['10_layer_activation'] = 'sigmoid'
            CNN_params['OPTIMIZER'] = 'rmsprop'
            CNN_params['LOSS'] = 'categorical_crossentropy'
            CNN_params['METRICS'] = 'accuracy'
            CNN_params['STEPS_PER_EPROCH'] = 100
            CNN_params['EPOCHS'] = 10
            CNN_params['VALIDATION_STEPS'] = 10
        
        return CNN_params

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        try:
            classifier.save(path)
        except Exception:
            self.log.error(traceback.format_exc())
            return False

        self.log.info(f'{path} is saved')
        return True


if __name__ == '__main__':
    cnn_model = CNN_Model()
    cnn_model.cnn(use_config = False, predict = True)