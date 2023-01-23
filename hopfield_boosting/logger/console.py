import json

from hopfield_boosting.logger.base import Logger


class ConsoleLogger(Logger):
    def log(self, log_dict: dict, epoch=None):
        print(json.dumps(log_dict))
