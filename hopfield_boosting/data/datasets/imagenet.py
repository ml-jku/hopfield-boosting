import os
import pickle

import numpy as np
import torch
from torchvision.transforms import ToPILImage


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

class ImageNet(torch.utils.data.Dataset):

    def __init__(self, transform=None, img_size=64, root=None):
        self.root = root
        self.S = np.zeros(11, dtype=np.int32)
        self.img_size = img_size
        self.labels = []
        for idx in range(1, 11):
            data_file = os.path.join(self.root, 'train_data_batch_{}'.format(idx))
            d = unpickle(data_file)
            y = d['labels']
            y = [i-1 for i in y]
            self.labels.extend(y)
            self.S[idx] = self.S[idx-1] + len(y)

        self.labels = np.array(self.labels)
        self.N = len(self.labels)
        self.curr_batch = -1

        self.transform = transform
        self.batch_cache = dict()

    def load_image_batch(self, batch_index):
        if batch_index in self.batch_cache:
            return self.batch_cache[batch_index]

        data_file = os.path.join(self.root, 'train_data_batch_{}'.format(batch_index))
        d = unpickle(data_file)
        x = d['data']
        
        img_size = self.img_size
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))

        self.batch_cache[batch_index] = x
        return x

    def get_batch_index(self, index):
        j = 1
        while index >= self.S[j]:
            j += 1
        return j

    def load_image(self, index):
        batch_index = self.get_batch_index(index)
        if self.curr_batch != batch_index:
            batch = self.load_image_batch(batch_index)
        
        return batch[index-self.S[batch_index-1]]

    def __getitem__(self, index):
        img = ToPILImage()(self.load_image(index))
        if self.transform is not None:
            img = self.transform(img)

        return img, index

    def __len__(self):
        return self.N