# MLFS - Machine Learning From Scratch


## About

This repository contains the Python implementation of some Machine Learning
algorithms using only `numpy` and `scipy`. The purpose is purely for
self-learning and the implementations do not focus on efficiency, but rather on
highlighting the inner workings.

Each model has a demo that can be run as follows:

    python -m <model> run

Run the following to see the possible parameters to set:

    python -m <model> run -- --help

For example:

    python -m mlfs.supervised.knn run


## Environment

    conda env create -f environment.yml


## Supervised

- [K-Nearest-Neighbors](mlfs/supervised/knn.py)
