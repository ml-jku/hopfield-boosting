import subprocess
from pathlib import Path
import os

def select_svhn_data(source_path='test_32x32.mat', dest_path='selected_test_32x32.mat'):
    # Code adopted from POEM: https://github.com/deeplearning-wisc/poem/blob/main/select_svhn_data.py
    import scipy.io as sio
    import os
    import numpy as np

    loaded_mat = sio.loadmat(source_path)

    data = loaded_mat['X']
    targets = loaded_mat['y']

    data = np.transpose(data, (3, 0, 1, 2))

    selected_data = []
    selected_targets = []
    count = np.zeros(11)

    for i, y in enumerate(targets):
        if count[y[0]] < 1000:
            selected_data.append(data[i])
            selected_targets.append(y)
            count[y[0]] += 1

    selected_data = np.array(selected_data)
    selected_targets = np.array(selected_targets)

    selected_data = np.transpose(selected_data, (1, 2, 3, 0))

    save_mat = {'X': selected_data, 'y': selected_targets}

    sio.savemat(dest_path, save_mat)


def download_dataset(dataset_url, download_path):
    subprocess.check_call(['wget', dataset_url, '-P', download_path])
    return Path(download_path) / dataset_url.split('/')[-1]


def extract_tar(tar_path, target_dir):
    if str(tar_path).endswith('.gz'):
        options = '-xvzf'
    else:
        options = '-xf'
    subprocess.check_call(['tar', options, tar_path, '-C', target_dir])

def remove_file(path):
    subprocess.check_call(['rm', path])


def prepare_all_datasets(download_path='downloaded_datasets/'):
    os.makedirs(download_path, exist_ok=True)

    tar_dataset_urls = [
        'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
        'http://data.csail.mit.edu/places/places365/test_256.tar',
        'https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz',
        'https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz',
        'https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz'
    ]

    for dataset_url in tar_dataset_urls:
        downloaded_path = download_dataset(dataset_url, download_path)
        extract_tar(downloaded_path, download_path)
        remove_file(downloaded_path)
        

    svhn_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

    svhn_test_file = download_dataset(svhn_url, Path(download_path) / 'svhn')
    select_svhn_data(svhn_test_file, Path(download_path) / 'svhn' / 'selected_test_32x32.mat')


if __name__ == '__main__':
    prepare_all_datasets()
