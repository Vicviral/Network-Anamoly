"""
Author:         Victor Loveday
Date:           01/08/2022   
"""

import logging

from src.timer import Singleton


class SetupLogger(metaclass=Singleton):

    def __init__(self):
        self._c_handler = logging.StreamHandler()
        self._f_handler = logging.FileHandler("run.log", mode="w")

        self._c_handler.setLevel(logging.INFO)
        self._f_handler.setLevel(logging.INFO)

        self.formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")

        self._c_handler.setFormatter(self.formatter)
        self._f_handler.setFormatter(self.formatter)

    @property
    def c_handler(self):
        return self._c_handler

    @property
    def f_handler(self):
        return self._f_handler


def setup_logger(logger):
    """
    Helper function to setup a logger with a file and console handler.

    :param logger: A logger from the source file looking to create the logger.
    :return: A logger with the correct file and console handler bound.
    """
    config = SetupLogger()

    logger.addHandler(config.c_handler)
    logger.addHandler(config.f_handler)

    logger.setLevel(logging.INFO)

    return logger
