import os
import time

import numpy as np
import scipy.io
import torch
from fastai.metrics import (
    accuracy,
    top_k_accuracy,
)
from fastai.vision import (
    ImageList,
    DatasetType,
    open_image,
    load_learner,
)


def get_learner(model_path, model_file, test_path, test_file):
    learn = load_learner(model_path, file=model_file, test=ImageList.from_csv(test_path, test_file, folder='//test'))
    return learn


def predict_one_image(model_path, model_file, test_path, test_file, img_path):
    img = open_image(img_path)
    learn = get_learner(model_path=model_path, model_file=model_file, test_path=test_path, test_file=test_file)
    pred_class, pred_idx, confidence = learn.predict(img)
    return pred_class, pred_idx.item(), confidence


def write_prediction(result_file, model_path, model_file, test_path, test_file, num_samples):
    learn = get_learner(model_path=model_path, model_file=model_file, test_path=test_path, test_file=test_file)
    out = open(result_file, 'a')
    start = time.time()

    for i in range(num_samples):
        filename = os.path.join(test_path, '%05d.jpg' % (i + 1))
        img = open_image(filename)
        pred_class, pred_idx, confidence = learn.predict(img)
        class_id = int(repr(pred_class).split(' ')[1])
        out.write('{}\n'.format(str(class_id)))

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()


def get_accuracy(model_path, model_file, test_path, test_file, test_annos_file, top_k):
    learn = get_learner(model_path=model_path, model_file=model_file, test_path=test_path, test_file=test_file)
    preds, y = learn.TTA(ds_type=DatasetType.Test)
    a = preds
    print(a.shape)
    labels = scipy.io.loadmat(test_annos_file)
    b = np.array(labels['annotations']['class'], dtype=np.int) - 1
    b = torch.from_numpy(b)
    acc = accuracy(a, b)
    top_k_acc = top_k_accuracy(a, b, k=top_k)
    return acc, top_k_acc
