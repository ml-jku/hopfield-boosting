from collections import defaultdict
from typing import Dict

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader


class OODEvaluator:
    def __init__(self, in_dataset, out_datasets, metrics, logger, device=None):
        self.in_dataset = in_dataset.loader
        self.out_datasets = {key: ds.loader for key, ds in out_datasets.items()}
        self.metrics = metrics
        self.logger = logger
        self.device = device

    def evaluate(self, ood_tester, epoch=None, prefix=None):
        if prefix is not None:
            prefix = f'{prefix}_'
        else:
            prefix = ''
        results = {}
        with torch.no_grad():
            in_scores = self.compute_scores_loader(self.in_dataset, ood_tester)
            for out_dataset_name, out_dataset in self.out_datasets.items():
                out_scores = self.compute_scores_loader(out_dataset, ood_tester)
                for metric_name, metric in self.metrics.items():
                    metric_result = metric(in_scores, out_scores)
                    results[f'{prefix}{metric_name}.{out_dataset_name}'] = metric_result

        if self.logger:
            self.logger.log(results, epoch=epoch)

        return metric_result
    
    def compute_scores_loader(self, loader: DataLoader, score_fn, num_samples=10000) -> Dict[float, dict]:
        total_samples = 0
        scores = []
        for xi, _ in loader:
            xi = xi.to(self.device)
            if total_samples >= num_samples:
                break
            total_samples += len(xi)
            score = score_fn(xi)
            scores.append(score)
        return torch.concat(scores, dim=0)

