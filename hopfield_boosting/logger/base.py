from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(self, log_dict: dict, epoch=None):
        pass
