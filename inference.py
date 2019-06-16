import os
import sys
import time

import numpy as np
import scipy.io
import torch
from fastai.metrics import accuracy
from fastai.vision import (
    ImageList,
    DatasetType,
    open_image,
    load_learner,
)


def get_class_names():
    """
    Get class names as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :return:
    """
    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    return class_names


def get_learner(model_path, model_file, test_path, test_file):
    """
    Loads the model learner from given model and test path and file.

    :param model_path: Path to dir where .pkl file is located.
    :param model_file: If multiple .pkl files are located in the same path, provide the exact model file name.
    :param test_path: Path to dir where test data is located
    :param test_file: Preprocessed test_labels.csv file, as was done in preprocess.py. It eases the fetching of ImageList.
    :return: The model learner.
    """
    learn = load_learner(model_path, file=model_file, test=ImageList.from_csv(test_path, test_file, folder='test'))
    return learn


def predict_one_image(learner, img_path, class_names):
    """
    Predicts top class for one image.

    :param learner: Model learner to classify images.
    :param img_path: Full absolute path to the image.
    :param class_names: Class names for cars as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :return: Class ID, Class name, and confidence score for the prediction.
    """
    img = open_image(img_path)
    pred_class, pred_idx, confidence = learner.predict(img)
    return pred_idx.item() + 1, class_names[pred_idx.item()][0][0], confidence[pred_idx.item()].item()


def get_accuracy(learner, test_annos_file):
    """
    Calculates the accuracy in percentage

    :param learner: Model learner to classify images
    :param test_annos_file: Test annotations file as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :return: A float number, which is accuracy in percentage
    """
    preds, y = learner.TTA(ds_type=DatasetType.Test)
    a = preds
    print(a.shape)
    labels = scipy.io.loadmat(test_annos_file)
    b = np.array(labels['annotations']['class'], dtype=np.int) - 1
    b = torch.from_numpy(b)
    acc = accuracy(a, b)
    return 100.0 * acc.item()


def write_prediction(result_file, learner, test_path, num_samples):
    """
    Writes the prediction to a file for the Stanford cars test dataset.

    :param result_file: User-defined result file name, which will have the class id for prediction.
    :param learner: Model learner to classify images.
    :param test_path: Path to dir where test images as located.
    :param num_samples: Test samples = 8041, as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :return:
    """
    out = open(result_file, 'a')
    start = time.time()

    for i in range(num_samples):
        filename = os.path.join(test_path, '%05d.jpg' % (i + 1))
        img = open_image(filename)
        pred_class, pred_idx, confidence = learner.predict(img)
        out.write('{}\n'.format(pred_idx.item() + 1))

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()


def write_prediction_with_score(result_file, learner, test_path, num_samples, class_names):
    """
    Writes the prediction to a file for the Stanford cars test dataset, including class name and confidence score.

    :param result_file: User-defined result file name, which will have the class id, class name and confidence score for prediction.
    :param learner: Model learner to classify images.
    :param test_path: Path to dir where test images as located.
    :param num_samples: Test samples = 8041, as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :param class_names: Class names for cars as provided by https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    :return:
    """
    out = open(result_file, 'a')
    start = time.time()
    out.write('class_id, class_name, confidence\n')

    for i in range(num_samples):
        filename = os.path.join(test_path, '%05d.jpg' % (i + 1))
        img = open_image(filename)
        pred_class, pred_idx, confidence = learner.predict(img)
        out.write('{}, {}, {}\n'.format(pred_idx.item() + 1, class_names[pred_idx.item()][0][0],
                                        confidence[pred_idx.item()].item()))

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()


if __name__ == '__main__':
    model_path = 'model'
    model_file = 'export-rn101_train_stage2-50e.pkl'
    test_path = 'data'
    test_file = 'test_labels.csv'
    test_annos_file = 'cars_test_annos_withlabels'
    learner = get_learner(model_path, model_file, test_path, test_file)
    acc = get_accuracy(learner, test_annos_file)
    print('The accuracy is {0}.'.format(acc))
    if 'with-confidence' in sys.argv:
        write_prediction_with_score(sys.argv[1], learner, 'data/test', 8041, get_class_names())
    else:
        write_prediction(sys.argv[0], learner, 'data/test', 8041)

    if 'predict-one' in sys.argv:
        pred_class, pred_idx, confidence = predict_one_image(learner, sys.argv[1], get_class_names())
    else:
        write_prediction(sys.argv[0], learner, 'data/test', 8041)
