# DALI AND PYTORCH PROFILER TUTORIAL

This repository contains code to accompany the tutorial.

The directory utils/ contains some useful logging code and resizing code that may also be helpful.

All original work contained in this repo is free to use according to the license provided.

All credited work is the property of the original authors.

## DALI INSTALLATION

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html


## RUNNING TENSORBOARD
from terminal:

`tensorboard --logdir <MAIN DIRECTORY> --port <DEFAULT:6006>`

## ADDITIONAL REQUIRMENTS

tensorboard for profiler requires:

`pip install tensorboard`

`pip install torch_tb_profiler`

torchaudio needs:

`conda install 'ffmpeg<5'`

`conda install -c conda-forge gcc`

tqdm can be installed via: `pip install tqdm`

## DATA

All three are sourced from Kaggle
1. Image data - [Imagenet1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
2. Audio data - [Birdclef](https://www.kaggle.com/competitions/birdclef-2022/data)

## BASE TUTORIALS
1. for audio - https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial