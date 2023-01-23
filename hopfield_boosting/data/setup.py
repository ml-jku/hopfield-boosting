import numpy as np
import torch


class DataSetup:
    def __init__(self,
                 dataset=None,
                 loader=None,
                 batch_sampler=None,
                 sampler=None,
                 wrapper=None,
                 split_sizes=None,
                 splits=None):
        self.dataset = dataset

        if wrapper is not None:
            self.dataset = wrapper(self.dataset)

        self.splits = splits
        self.split_sizes = split_sizes

        if splits is not None:
            assert loader is None and batch_sampler is None and sampler is None
            assert np.allclose(sum(split_sizes.values()), 1.), 'Relative size of splits does not add up to 1.'

            split_lens = {key: int(value * len(dataset)) for key, value in list(split_sizes.items())[:-1]}
            split_lens[list(split_sizes.keys())[-1]] = len(dataset) - sum(split_lens.values())

            generator = torch.Generator()
            generator.manual_seed(42)
            subsets = {key: subset for key, subset in zip(split_sizes.keys(), torch.utils.data.random_split(dataset, split_lens.values(), generator=generator))}

            for split_name in self.splits.keys():
                subset = subsets[split_name]
                self.__dict__[split_name] = self.splits[split_name](dataset=subset)

        else:
            if sampler:
                self.sampler = sampler(self.dataset)
            else:
                self.sampler = None
            if batch_sampler:
                self.batch_sampler = batch_sampler(self.dataset)
            else:
                self.batch_sampler = None
            self.collate_fn = self.batch_sampler.collate_fn if self.batch_sampler else None
            self.loader = loader(dataset=self.dataset, sampler=self.sampler, batch_sampler=self.batch_sampler, collate_fn=self.collate_fn)
