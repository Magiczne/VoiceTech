import configparser
from os import path

from .Singleton import Singleton


class Config(metaclass=Singleton):
    def __init__(self):
        self.config_ = configparser.ConfigParser()

        self.config_.read(path.abspath('config.ini'))

    def get_param(self, section, param, dtype):
        """
        Get param from the config file

        :raises         KeyError
        :param section: Section name
        :param param:   Param name
        :param dtype:   Param data type
        :return:        Config value
        """
        if section not in self.config_:
            raise KeyError('Section not found in the config file')

        if param not in self.config_[section]:
            raise KeyError('Param not found in the config file')

        return dtype(self.config_[section][param])
