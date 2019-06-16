# aiforsea-cv-cars


This repository is to do car recognition by transfer learning, fine-tuning ResNet models with Cars Dataset from Stanford.

## Summary

- Trained and tested with ResNet50, Resnet101, ResNet152.
- Ceteris paribus, a model with denser networks performed better. But, there is a trade-off between resources and accuracy. So, the goal of this project is to arrive at a reasonable model (say > 90% accuracy) within a reasonable aount of training time. Given more time and resources, we could definitely fine-tune more to improve accuracy.
- With this goal in mind, I could come with 90.8% accuracy on validation set and 91.4% on test set by fine-tuning ResNet101. Specifically, I used a differential learning rate between `slice()` over 40 epochs.
- Recent state-of-the-art results on the Stanford Cars-196 data set published so far:  


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Fastai](https://github.com/fastai/fastai/blob/master/README.md#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- Environment:
```bash
$ nvidia-smi
Sat Jun 15 06:42:17 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   53C    P0    24W /  75W |   5953MiB /  7611MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      8503      C   /opt/anaconda3/bin/python                   5943MiB |
+-----------------------------------------------------------------------------+
```

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd aiforsea-cv-cars
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```