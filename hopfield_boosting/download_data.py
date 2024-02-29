import subprocess
from pathlib import Path
import os

from dotenv import load_dotenv

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


def download_dataset(dataset_url, download_path, supress_stderr=False):
    if supress_stderr:
        subprocess.check_call(['wget', dataset_url, '-P', download_path], stderr=subprocess.DEVNULL)
    else:
        subprocess.check_call(['wget', dataset_url, '-P', download_path])
    return Path(download_path) / dataset_url.split('/')[-1]


def extract_tar(tar_path, target_dir, supress_output=False):
    os.makedirs(target_dir, exist_ok=True)
    if str(tar_path).endswith('.gz'):
        options = '-xvzf'
    else:
        options = '-xf'
    if supress_output:
        subprocess.check_call(['tar', options, tar_path, '-C', target_dir], stdout=subprocess.DEVNULL)
    else:
        subprocess.check_call(['tar', options, tar_path, '-C', target_dir])

def remove_file(path):
    subprocess.check_call(['rm', path])


def prepare_all_datasets(download_path='downloaded_datasets/', supress_output=False):
    os.makedirs(download_path, exist_ok=True)

    tar_dataset_urls = {
        'TEXTURES': 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
        'PLACES': 'http://data.csail.mit.edu/places/places365/test_256.tar',
        'LSUN_CROP': 'https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz',
        'LSUN_RESIZE': 'https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz',
        'ISUN': 'https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz'
    }

    for dataset_name, dataset_url in tar_dataset_urls.items():
        print(f'Downloading {dataset_name}')
        downloaded_path = download_dataset(dataset_url, download_path, supress_stderr=supress_output)
        if dataset_name == 'PLACES':
            extract_path = Path(download_path) / 'places365'
            os.makedirs(extract_path)
        else:
            extract_path = download_path
        print(f'Extracting {dataset_name}')
        extract_tar(downloaded_path, extract_path, supress_output=supress_output)
        remove_file(downloaded_path)


    svhn_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

    print('Downloading SVHN')
    svhn_test_file = download_dataset(svhn_url, Path(download_path) / 'svhn')
    print('Selecting SVHN')
    select_svhn_data(svhn_test_file, Path(download_path) / 'svhn' / 'selected_test_32x32.mat')


if __name__ == '__main__':
    load_dotenv()
    download_path = os.getenv('DOWNLOADED_PATH', 'downloaded_datasets')
    prepare_all_datasets(download_path)
