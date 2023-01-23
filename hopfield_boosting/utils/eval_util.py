import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance



def eval_auroc(a, b):
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))], axis=0)
    score = np.concatenate([a, b], axis=0)

    auroc = roc_auc_score(labels, score)

    return auroc


def eval_metrics_str(a, b):
    result_str = f"AUROC: {eval_auroc(a, b):.4f}\n" + f"Wasserstein distance: {wasserstein_distance(a, b):.4f}"

    return result_str


def dump_eval_metrics(a, b):
    result_str = eval_metrics_str(a, b)
    print(result_str)