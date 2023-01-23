from pathlib import Path
import torch
from torch import nn
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

# helper function to load a model
def model_file(model_path, epoch=99, name='model'):
    model_file = model_path / str(epoch) / (name + '.ckpt')

    return model_file


def load_model(model_file, map_location='cpu'):
    model = torch.load(model_file, map_location=map_location)

    return model


def create_model(model_path, model_config='cfg.yaml', device='cuda'):
    with open(model_path / model_config) as p:
        config = OmegaConf.create(yaml.safe_load(p))
    
    resnet = instantiate(config['model'])
    projection_head = instantiate(config['projection_head'])
    classifier = nn.Linear(512, config.num_classes)

    return config, resnet.eval().to(device), projection_head.eval().to(device), classifier.eval().to(device)


def load_model_weights(model_path, resnet, projection_head, classifier, epoch='', device='cuda'):
    resnet.load_state_dict(load_model(model_file(model_path, epoch=epoch)))
    projection_head.load_state_dict(load_model(model_file(model_path, name='projection_head', epoch=epoch)))
    classifier.load_state_dict(load_model(model_file(model_path, name='classifier', epoch=epoch)))
    
    return resnet.eval().to(device), projection_head.eval().to(device), classifier.eval().to(device)