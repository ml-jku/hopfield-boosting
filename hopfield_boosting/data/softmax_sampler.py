import torch


class SoftmaxBorderSampler:
    def __init__(self, num_batches, num_tested_samples, beta, out_batch_size, replacement, device):
        self.num_batches = num_batches
        self.num_tested_samples = num_tested_samples
        self.beta = beta
        self.out_batch_size = out_batch_size
        self.num_samples = self.num_batches * self.out_batch_size
        self.replacement = replacement
        self.device = device

    def sample_border_points(self, energy_fn, data):

        energies = []
        idxs = []

        num_tested_samples = 0

        with torch.no_grad():

            for x, idx in data:
                if num_tested_samples >= self.num_tested_samples:
                    break
                energy = energy_fn(x.to(self.device))

                energies.append(energy.cpu())
                idxs.append(idx)

                num_tested_samples += len(x)

            energies = torch.concat(energies, dim=0)
            idxs = torch.concat(idxs, dim=0)

            p = torch.softmax(-self.beta*energies, dim=-1)
            samples = torch.multinomial(p, self.num_samples, replacement=self.replacement)
            sample_idxs = idxs[samples]

        return torch.utils.data.DataLoader(torch.utils.data.Subset(data.dataset, sample_idxs), shuffle=True, batch_size=self.out_batch_size)
