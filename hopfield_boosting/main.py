import os
import random
from pathlib import Path
from contextlib import contextmanager

import hydra
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

import hopfield_boosting
import wandb
from hopfield_boosting.util import infer_loader


@contextmanager
def get_patterns(model, in_loader, aux_loader, device):
    with infer_loader(in_loader, model=model, device=device) as id, \
        infer_loader(aux_loader, model=model, device=device, max_samples=len(in_loader)*in_loader.batch_size) as aux:
        yield id, aux


def evaluate_id_classifier(classifier, val_loader, criterion, device, epoch):
    logits_val = []
    ys_val = []
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(device), y_val.to(device)

        logit_val = classifier(x_val)

        logits_val.append(logit_val)
        ys_val.append(y_val)
    
    logits_val = torch.concat(logits_val, dim=0)
    ys_val = torch.concat(ys_val, dim=0)

    loss = criterion(logits_val, ys_val)
    acc = torch.mean((torch.argmax(logits_val, dim=-1) == ys_val).to(torch.float))

    d = {'classifier/loss_val': loss,
         'classifier/acc_val': acc,
         'general/epoch': epoch}
    return d


def evaluate(id, aux, trainer, data, device, ood_evaluators, epoch):
    with torch.no_grad():
        log_dict = evaluate_id_classifier(nn.Sequential(trainer.model, trainer.classifier), data.no_aug.val.loader, trainer.criterion, device, epoch=epoch)
        if epoch % 25 == 24:
            ood_tester = trainer.ood_tester(emb_in=id, emb_out=aux)
            for odd_evaluator in ood_evaluators.values():
                odd_evaluator.evaluate(
                    ood_tester,
                    epoch=epoch
                )
    return log_dict


def select_aux_data(id, aux, trainer, raw_aux_loader, sampler):
    if sampler:
        border_energy = trainer.energy_fn(a=id, b=aux)
        border_energy = nn.Sequential(trainer.model, trainer.projection_head, border_energy)
        aux_loader_selected = sampler.sample_border_points(energy_fn=border_energy, data=raw_aux_loader)
    else:
        aux_loader_selected = raw_aux_loader

    return aux_loader_selected


def evaluate_and_select_aux_data(trainer, data, aux_loader, device, ood_evaluators, sampler, early_stopping, run_dir, epoch):
    trainer.eval()
    projection_model = nn.Sequential(trainer.model, trainer.projection_head)
    with get_patterns(projection_model, data.aug.train.loader, aux_loader, device) as (id, aux):
        log_dict = evaluate(
            id=id,
            aux=aux,
            trainer=trainer,
            data=data,
            device=device,
            ood_evaluators=ood_evaluators,
            epoch=epoch
        )
        aux_loader_selected = select_aux_data(
            id=id,
            aux=aux,
            trainer=trainer,
            raw_aux_loader=aux_loader,
            sampler=sampler
        )

    torch.save(trainer.model.state_dict(), f'{run_dir}/model.ckpt')
    torch.save(trainer.classifier.state_dict(), f'{run_dir}/classifier.ckpt')
    torch.save(id, f'{run_dir}/id.ckpt')
    torch.save(aux, f'{run_dir}/ood.ckpt')
    try:
        torch.save(trainer.projection_head.state_dict(), f'{run_dir}/projection_head.ckpt')
    except:
        print('Projection head could not be saved!')

    wandb.log(log_dict)
    if early_stopping.should_stop(log_dict):
        return
    trainer.train()
    return aux_loader_selected


def torch_deterministic():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@hydra.main(config_path='config', config_name='resnet-18-cifar-10-aux-from-scratch', version_base=None)
def main(config: DictConfig):

    print(f'Location of hopfield_boosting library: {hopfield_boosting.__file__}')

    if config.use_seed:
        torch.cuda.random.manual_seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    if config.force_deterministic:
        torch_deterministic()

    project_name = config.project_name

    wandb.init(project=project_name, config={'hydra': OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True)})
    wandb.define_metric('epoch')

    print(OmegaConf.to_yaml(config))

    run_dir = Path(config.project_root) / 'runs' / project_name / wandb.run.id
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(config, f'{run_dir}/cfg.yaml', resolve=True)

    device = config.device
    trainer = instantiate(config.trainer).to(device)
    model = trainer.model
    classifier = trainer.classifier
    projection_head = trainer.projection_head
    opt = trainer.optim
    early_stopping = instantiate(config.early_stopping)

    beta = config.beta

    data = instantiate(config.data)
    aux_data = instantiate(config.auxiliary)
    aux_loader = aux_data.loader

    sampler = instantiate(config.sampler, num_batches=len(data.aug.train.loader), beta=beta)
    ood_evaluators = instantiate(config.ood_eval)


    if config.get('scheduler'):
        if 'T_max' in config.scheduler.keys():
            scheduler = instantiate(config.scheduler, optimizer=opt, T_max=len(data.aug.train.loader) * config.no_epochs)
        else:
            scheduler = instantiate(config.scheduler, optimizer=opt)
    else:
        scheduler = None

    wandb.watch([model, classifier, projection_head], log_freq=100, log='all')

    print('Initial Evaluation')
    aux_loader_selected = evaluate_and_select_aux_data(
        trainer=trainer,
        data=data,
        aux_loader=aux_loader,
        device=device,
        ood_evaluators=ood_evaluators,
        sampler=sampler,
        early_stopping=early_stopping,
        run_dir=run_dir,
        epoch=-1
    )

    for epoch in tqdm(range(config.no_epochs)):
        
        in_losses = []
        for i, ((xs_train, ys_train), (xs_ood, _)) in enumerate(zip(data.aug.train.loader, aux_loader_selected)):
            trainer.train()

            log_dict = {
                'general/beta': beta,
                'general/lr': opt.param_groups[0]['lr'],
                'general/epoch': epoch
            }

            xs_train, ys_train = xs_train.to(device), ys_train.to(device)
            xs_ood = xs_ood.to(device)

            loss, info_dict = trainer.step(xs_train, ys_train, xs_ood)
            in_losses.append(info_dict['classifier/in_loss'])

            log_dict.update(info_dict)

            wandb.log(log_dict)

            if scheduler and isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

        if config.recursive_sampling:
            aux_loader_selected = aux_loader_selected
        else:
            aux_loader_selected = aux_loader

        aux_loader_selected = evaluate_and_select_aux_data(
            trainer=trainer,
            data=data,
            aux_loader=aux_loader_selected,
            device=device,
            ood_evaluators=ood_evaluators,
            sampler=sampler,
            early_stopping=early_stopping,
            run_dir=run_dir,
            epoch=epoch
        )

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(torch.mean(torch.tensor(in_losses)))
            elif not isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

    wandb.finish(0)

if __name__ == '__main__':
    main()
