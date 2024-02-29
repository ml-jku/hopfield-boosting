# Energy-based Hopfield Boosting for Out-of-Distribution Detection

[![arXiv](https://img.shields.io/badge/arXiv-2306.14884-b31b1b.svg)](TODO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of "Energy-based Hopfield Boosting for Out-of-Distribution Detection". The paper is available [here TODO](TODO).

https://github.com/claushofmann/hopfield-classifier/assets/23155858/83021bdf-755b-4e79-8b52-4dc999cbc56f

## Installation

- Hopfield Boosting works best with Anaconda ([download here](https://www.anaconda.com/download)). 
  To install Hopfield Boosting and all dependencies, run the following commands:

  ```
  conda env create -f environment.yml
  conda activate hopfield-boosting
  pip install -e .
  ```

## Weights and Biases

- Hopfield Boosting supports logging with Weights and Biases (W&B). By default, W&B will log all metrics in [anonymous mode](https://docs.wandb.ai/guides/app/features/anon). Note that runs logged in anonymous mode will be deleted after 7 days. To keep the logs, you need to [create a W&B account](https://docs.wandb.ai/quickstart). When done, login to your account using the command line.

## Data Sets
To run, you need the following data sets. We follow the established benchmark, which is also used by e.g. [Lui et al. (2020)](https://arxiv.org/abs/2010.03759) and [Ming et al. (2022)](https://arxiv.org/abs/2206.13687).

### In-Distribution Data Sets

  * [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html): Automatically downloaded by PyTorch

### Auxiliary Outlier Data Set

  * [ImageNet-RC](https://patrykchrabaszcz.github.io/Imagenet32/): We use ImageNet64x64, which can be downloaded from the [ImageNet Website](http://image-net.org/download-images).


### Out-of-Distribution Validation Data Sets

* **MNIST**: Automatically downloaded by PyTorch
* **FashionMNIST**: Automatically downloaded by PyTorch

### Out-of-Distribution Test Data Sets

The OOD test data is comprised of a selection of vision data sets:

* **SVHN**: Street View House Numbers
* **Places 365**: Scene recognition data set
* **LSUN-Resize**: A resized version of the Large-scale Scene UNderstanding Challenge
* **LSUN-Crop**: A cropped version of the Large-scale Scene UNderstanding Challenge
* **iSUN**: Contains a large number of different scenes
* **Textures**: A collection of textural images in the wild

We have included a Python script that conveniently downloads all OOD Test data sets. To execute it, simply run

```
python -m hopfield_boosting.download_data
```

The downloaded data sets will be placed in the currently active directory under `downloaded_datasets/`.

## How to Run

- Set the paths to the data sets: Copy the `.env.examples` file located in the root directory of the repository. 
  Name the newly created file `.env`. 
  Customize the new file to contain the paths to the data sets on your machine.
  You can also set a `project_root`, which is where Hopfield Boosting will store your model checkpoints.

- To run Hopfield Boosting on CIFAR-10, run the command
  ```
  python -m hopfield_boosting -cn resnet-18-cifar-10-aux-from-scratch
  ```

- For CIFAR-100, use the command
  ```
  python -m hopfield_boosting -cn resnet-18-cifar-100-aux-from-scratch
  ```

- The performance on the OOD validation data sets will be logged to W&B; the performance on the OOD test sets will be logged to a file located in
  `test_logs` named according to the `run.id` from W&B.

## 📓 Demo Notebook

We have provided a demo notebook [here](notebooks/hopfield_boosting_demo_extended.ipynb) where we demonstrate the capability of Hopfeild Boosting to detect OOD inputs. We provide a pre-trained model trained on CIFAR-10 for running the notebook, which is available for download [here](https://drive.google.com/file/d/1LK1VyjvQfA3qUG8LGue0IBOy0Sja2GJb/view?usp=sharing).

To run, first set the paths to the data sets and the model in `hopfield_boosting_notebook_config.yaml`. The notebook uses additional data sets. You can find the link to download these data sets in the notebook itself.


# 📚 Citation

If you found this repository helpful, consider giving it a ⭐ and cite our paper:

```
@article TODO
```
