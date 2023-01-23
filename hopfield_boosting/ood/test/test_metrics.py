import pytest
import torch
import numpy as np

from hopfield_boosting.ood.metrics import AUROCOODMetric, AUPRSOODMetric, FPR95OODMetric

#@pytest.fixture
def dummy_ood_scores_1():
    in_scores = torch.tensor([.21, .31, .41, .51])
    out_scores = torch.tensor([.0, .1, .2, .3, .4])

    return in_scores, out_scores

def dummy_ood_scores_2():
    in_scores = torch.tensor([1.1, 1.2, 1.3])
    out_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return in_scores, out_scores

@pytest.mark.parametrize('data_expected_result', [(dummy_ood_scores_1, 0.85), (dummy_ood_scores_2, 1.0)])
def test_auroc(data_expected_result):
    data, expected_result = data_expected_result
    in_scores, out_scores = data()
    auroc = AUROCOODMetric()
    assert np.allclose(auroc(in_scores, out_scores), expected_result)


@pytest.mark.parametrize('data_expected_result', [(dummy_ood_scores_1, 0.8542), (dummy_ood_scores_2, 1.0)])
def test_auprs(data_expected_result):
    data, expected_result = data_expected_result
    in_scores, out_scores = data()
    auprs = AUPRSOODMetric()
    assert np.allclose(auprs(in_scores, out_scores), expected_result, atol=1e-4)

@pytest.mark.parametrize('data_expected_result', [(dummy_ood_scores_1, 0.4), (dummy_ood_scores_2, 0.0)])
def test_fpr95(data_expected_result):
    data, expected_result = data_expected_result
    in_scores, out_scores = data()
    fpr95 = FPR95OODMetric()
    assert np.allclose(fpr95(in_scores, out_scores), expected_result)
