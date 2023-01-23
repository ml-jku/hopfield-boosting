import pytest
import torch
from torch import nn

from hopfield_boosting.energy import HopfieldClassifier
from hopfield_boosting.classification_head import SimpleHopfieldClassificationHead
from hopfield_boosting.ood import MSPOODDetector, EnergyOODDetector, MaxLogitOODDetector, ComposedOODDetector


@pytest.fixture
def in_data():
    in_data = torch.rand([100, 100])
    in_stored, in_state = in_data[:50], in_data[50:]
    in_stored_y = torch.concat([torch.zeros([25]), torch.ones([25])]).to(torch.int64)
    return in_stored, in_stored_y, in_state

@pytest.fixture
def out_data():
    return -torch.rand([50, 100])

@pytest.fixture
def classifier(in_data):
    with HopfieldClassifier(nn.Identity(), SimpleHopfieldClassificationHead(1.)).store_tensor(in_data[0], y=in_data[1]) as cls:
        yield cls


@pytest.fixture
def detectors(classifier):
    return [EnergyOODDetector(), MaxLogitOODDetector(), MSPOODDetector()]


def test_detectors(detectors, in_data, out_data, classifier):
    _, _, in_data = in_data
    for detector in detectors:
        in_scores = next(iter(detector.compute_scores(in_data, classifier, beta=1.).values()))
        out_scores = next(iter(detector.compute_scores(out_data, classifier, beta=1.).values()))
        if not isinstance(detector, MSPOODDetector): 
            # MSP unfortunately cant detect those outliers
            assert torch.min(in_scores) > torch.max(out_scores)


def test_composed_detector(detectors, classifier, in_data, out_data):
    _, _, in_data = in_data
    detector = ComposedOODDetector(detectors)

    in_scores = detector.compute_scores(in_data, classifier, beta=1.)
    out_scores = detector.compute_scores(out_data, classifier, beta=1.)

    for k, in_score in in_scores.items():
        out_score = out_scores[k]
        if not k == 'msp':
            assert torch.min(in_score) > torch.max(out_score)
