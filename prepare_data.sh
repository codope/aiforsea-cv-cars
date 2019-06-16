#!/usr/bin/env bash

echo "Downloading the datasets"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat"
wget --no-check-certificate "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"

echo "Preparing the datasets"
python preprocess.py
