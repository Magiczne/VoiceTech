import configparser


class Config:
    def __init__(self):
        self.config_ = configparser.ConfigParser()
        self.config_.read('../config.ini')

    def get_param(self, section, param):
        """
        Get param from the config file

        :raises
        :param section: Section name
        :param param:   Param name
        :return:        Config value
        """
        if section not in self.config_:
            raise KeyError('Section not found in the config file')

        if param not in self.config_[section]:
            raise KeyError('Param not found in the config file')

        return self.config_[section][param]
