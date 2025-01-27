
This repository is the official implementation of "Jensen-Shannon Divergence Based Novel Loss Functions for Bayesian Neural Networks".

## Requirements

To install requirements:

The code is implemented in MxNet 1.9.1 with Python3. The code can run on both GPUs and CPUs. 

The exact GPU enabled MxNet docker image can be downloaded from (https://hub.docker.com/layers/mxnet/python/1.9.1_gpu_cu110_py3/images/sha256-3f33fdb2fa8f1bb1840d47122608d5b938d6f6a76caedf35172212e2ba5709f5)

A few standard additional packages such as tqdm may be required.


Datasets:

The experiments in the paper conducted on two classification datasets are presented here:

1) CIFAR-10 dataset: This can be downloaded from the source at (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz). Extract the data after download.

2) Histopathology dataset: The exact version of the dataset used in the experiments with the preprocessing and the train-test split can be downloaded from (https://osf.io/mg65a/?view_only=6ec96fc2fa4a435e9a431783575ca1e8)

The datasets should be copied to the "data" folder. The data folder consists of two subfolders: "cifar" and "histo". Place the datasets in their respective folders before running the experiments. 


## Training

config.py file can be modified to run different experiments of the paper. This file contains all the hyper-parameters and the setup to run the experiments.

To train the model(s) in the paper, run this command:

```train
python3 main.py
```

## Evaluation

The results of training, evaluation, and testing are stored in the "Experiments" folder.

## Acknowledgments

This work was supported by grant DE-SC0023432 funded by the U.S. Department of Energy, Office of Science. This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231, using NERSC awards BES-ERCAP0025205 and BES-ERCAP0025168.


