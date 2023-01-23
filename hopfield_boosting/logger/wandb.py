import wandb
from hopfield_boosting.logger.base import Logger


class WandbLogger(Logger):
    def __init__(self, prefix='ood/'):
        self.prefix = prefix

    def log(self, log_dict: dict, epoch=None):
        if self.prefix is not None:
            log_dict = {f'{self.prefix}{key}': value for key, value in log_dict.items()}
        log_dict['general/epoch'] = epoch
        wandb.log(log_dict)
