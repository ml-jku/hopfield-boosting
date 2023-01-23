import json
import os

import wandb

from hopfield_boosting.logger.base import Logger


class FileLogger(Logger):
    def __init__(self, path: str, mode='w'):
        os.makedirs(path, exist_ok=True)
        self.file_path = os.path.join(path, f'{wandb.run.id}.txt')
        assert mode in ('w', 'a'), 'Please choose either mode "w" or "a"'
        self.mode = mode

    def log(self, log_dict: dict, epoch=None):
        with open(self.file_path, self.mode) as f:
            f.write(f'{json.dumps(log_dict, indent=2)}\n')
