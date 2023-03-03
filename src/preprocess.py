import configparser
import os

from logger import Logger

SHOW_LOG = True



class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.getcwd()
        self.data_path = os.path.join(self.project_path, 'data')
        self.train_path = os.path.join(self.data_path, 'train')
        self.test_path = os.path.join(self.data_path, 'test')
        self.log.info('DataMaker is ready')

    def preprocessing_data(self) -> bool:
        if not os.path.exists(os.path.join(self.train_path, 'cat')):
            os.mkdir(os.path.join(self.train_path, 'cat'))
        if not os.path.exists(os.path.join(self.train_path, 'dog')):
            os.mkdir(os.path.join(self.train_path, 'dog'))
        for file in os.listdir(self.train_path):
            if file[-3:] == 'jpg':
                if file[:3] == 'cat':
                    os.replace(os.path.join(self.train_path, file), os.path.join(self.train_path, 'cat', file))
                else:
                    os.replace(os.path.join(self.train_path, file), os.path.join(self.train_path, 'dog', file))
        if ((len(os.listdir(os.path.join(self.train_path, 'cat'))) > 0) and 
            (len(os.listdir(os.path.join(self.train_path, 'dog'))) > 0)):
            self.log.info('Train data is ready')
            self.config['DATA'] = {'train_data': self.train_path,
                                   'test_data': self.test_path}
            self.config['PREPROCESSED_DATA'] = {'train_data': self.train_path,
                                                'test_data': self.test_path}
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
            return True
        else:
            self.log.error('Train data is not ready')
            return False

if __name__ == '__main__':
    data_maker = DataMaker()
    data_maker.preprocessing_data()