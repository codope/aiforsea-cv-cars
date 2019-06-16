#!/usr/bin/env bash

echo "Downloading the datasets"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
wget -c "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat"
wget --no-check-certificate "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"

mkdir model
cd model
wget -c "https://github.com/codope/aiforsea-cv-cars/raw/master/export-rn101_train_stage2-50e.pkl"
cd ../

echo "Preparing the datasets"
python preprocess.py
