from abc import ABC, abstractmethod

class EarlyStopping(ABC):
    @abstractmethod
    def should_stop(log_dict) -> bool:
        pass


class LowAccuracyEarlyStopping(ABC):
    def __init__(self, accuracy, epoch, key='classifier/acc_val', epoch_key='general/epoch'):
        self.accuracy = accuracy
        self.epoch = epoch
        self.key = key
        self.epoch_key = epoch_key

    def should_stop(self, log_dict):
        if self.epoch == log_dict[self.epoch_key]:
            return log_dict[self.key] < self.accuracy
        else:
            return False
