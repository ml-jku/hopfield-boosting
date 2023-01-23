from abc import ABC, abstractmethod


import torch
from torch import nn

from hopfield_boosting.energy import OneSidedEnergy
from hopfield_boosting.util import Negative, FirstElement

class Trainer(nn.Module, ABC):
    def __init__(self, model, classifier, criterion, optim, energy_fn, ood_weight, projection_head=None, beta=None, use_ood=True):
        super(Trainer, self).__init__()
        self.model = model
        self.classifier = classifier
        self.criterion = criterion
        self.optim = optim(params=nn.Sequential(model, classifier).parameters())
        self.energy_fn = energy_fn
        self.ood_weight = ood_weight
        self.beta = beta
        self.use_ood = use_ood
        if projection_head is None:
            projection_head = nn.Identity()
        self.projection_head = projection_head

    def forward(self, x, y, x_aux):
        assert x_aux is not None
        len_xs_ood = len(x_aux)
        if self.use_ood:
            xs_train = torch.concat([x, x_aux], dim=0)
        else:
            xs_train = x

        emb = self.model(xs_train)
        emb_projected = self.projection_head(emb)
        logits = self.classifier(emb)

        if torch.any(torch.isnan(logits)):
            assert False

        logits_in = logits[:len(x)]
        logits_out = logits[len(x):]

        in_loss = self.criterion(logits_in, y)
        pred = torch.argmax(logits_in, dim=-1)
        acc = torch.mean((pred == y).to(torch.float))
        loss = in_loss

        if self.use_ood:

            ood_loss, ood_dict = self.ood_loss(emb_projected[:-len_xs_ood], emb_projected[-len_xs_ood:], logits_in, logits_out)
            loss = loss + self.ood_weight * ood_loss

        else:
            ood_loss = torch.tensor(0.)
            ood_dict = {}

        info_dict = {
            'classifier/acc': acc,
            'classifier/in_loss': in_loss,
            'classifier/in_logits': logits_in,
            'general/loss': loss,
            'out/ood_loss': ood_loss,
        }

        info_dict.update({f'energies/{k}': v for k, v in ood_dict.items() if not k == 'attn'})

        info_dict = {k: v.detach() for k, v in info_dict.items()}

        return loss, info_dict
    
    def step(self, x, y, x_aux):
        loss, info_dict = self(x, y, x_aux)

        self.optim.zero_grad()
        loss.backward()

        self.optim.step()

        return loss, info_dict
    
    @abstractmethod
    def ood_loss(self, emb_in, emb_out, logits_in, logits_out):
        pass

    @abstractmethod
    def ood_tester(self, emb_in, emb_out) -> float:
        """
        OOD Tester. The output should implement the following convention: High -> ID; Low -> OOD.
        """
        pass


class HopfieldTrainer(Trainer):
    def ood_loss(self, emb_in, emb_out, logits_in, logits_out):
        border_energy = self.energy_fn(a=emb_in, b=emb_out)
        energies, info_dict = border_energy(torch.concat([emb_in, emb_out], dim=0), return_dict=True)
        return -torch.mean(energies), {'border_energies': energies, **info_dict}

    def ood_tester(self, emb_in, emb_out):
        energy_one_sided = OneSidedEnergy(a=emb_in, b=emb_out, beta_a=self.beta, beta_b=self.beta, normalize=True)
        
        return nn.Sequential(self.model, self.projection_head, energy_one_sided, Negative())


class EnergyTrainer(Trainer):  # implementation of https://arxiv.org/pdf/2010.03759.pdf
    def __init__(self, model, classifier, criterion, optim, m_in=-23, m_out=-5, energy_fn=None, ood_weight=0.1):  # hyperparameters for CIFAR-10 taken from https://arxiv.org/pdf/2010.03759.pdf
        super().__init__(model, classifier, criterion, optim, energy_fn, ood_weight)
        self.m_in = m_in
        self.m_out = m_out

    def ood_loss(self, emb_in, emb_out, logits_in, logits_out):
        in_energy = -torch.logsumexp(logits_in, dim=-1)
        out_energy = -torch.logsumexp(logits_out, dim=-1)
        in_losses = torch.maximum(torch.tensor(0.), in_energy - self.m_in)**2
        out_losses = torch.maximum(torch.tensor(0.), self.m_out - out_energy)**2
        loss = torch.mean(in_losses) + torch.mean(out_losses)
        return loss, {'in_energy': in_energy, 'out_energy': out_energy, 'in_losses': in_losses, 'out_losses': out_losses}

    def ood_tester(self, emb_in, emb_out):
        class Energy(nn.Module):
            def forward(self, logits):
                return -torch.logsumexp(logits, dim=-1)
        return nn.Sequential(self.model, self.classifier, Energy(), Negative())


class MSPTrainer(Trainer):  # implementation of https://arxiv.org/pdf/1812.04606.pdf
    def __init__(self, model, classifier, criterion, optim, energy_fn=None, ood_weight=0.5):
        super().__init__(model, classifier, criterion, optim, energy_fn, ood_weight)

    def ood_loss(self, emb_in, emb_out, logits_in, logits_out):
        return -(logits_out.mean(-1) - torch.logsumexp(logits_out, dim=-1)).mean(), {}

    def ood_tester(self, emb_in, emb_out):
        class MSP(nn.Module):
            def __init__(self) -> None:
                super(MSP, self).__init__()

            def forward(self, logits):
                return torch.max(torch.softmax(logits, dim=-1), dim=-1)[0]
        return nn.Sequential(self.model, self.classifier, MSP())
