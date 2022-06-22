import os
import torch
import torch.nn as nn
import numpy as np
from multi_train_utils.test_utils import group_image, _preprocess, cosin_metric
from config import Config
from models.fmobilenet import FaceMobileNet
import pandas as pd
import time
import datetime

def label_dict(data_path):

    lbl_dict = {}
    person_list = sorted(os.listdir(data_path), key=str.lower)
    for i, p in enumerate(person_list):
        img_num = len(os.listdir(data_path + '/' + p))
        lbl_dict[p] = (i, img_num)

    return lbl_dict

def all_data_featurize(data_path, conf, model):

    img_list = []
    person_list = sorted(os.listdir(data_path), key=str.lower)
    for p in person_list:
        person_path = data_path + '/' + p
        imgs = os.listdir(person_path)

        for i in imgs:
            img_path = person_path + '/' + i
            img_list.append(img_path)

    groups = group_image(img_list, conf.test_batch_size)
    feature_dict = dict()
    for group in groups:
        data = _preprocess(group, conf.test_transform)
        with torch.no_grad():
            features = model(data)
            features = features.cpu().numpy()
        d = {img: [feature, img.split('/')[-2]] for (img, feature) in zip(group, features)}
        feature_dict.update(d)

    return feature_dict

def test_data_featurize(test_path, conf, model):

    img_list = []
    imgs = sorted(os.listdir(test_path), key=lambda x: int(x.split('.')[0]))
    # imgs = os.listdir(test_path)
    for i in imgs:
        img_path = test_path + '/' + i
        img_list.append(img_path)

    groups = group_image(img_list, conf.test_batch_size)
    feature_dict = dict()
    for group in groups:
        data = _preprocess(group, conf.test_transform)
        with torch.no_grad():
            features = model(data)
            features = features.cpu().numpy()
        d = {img.split('/')[-1].split('.')[0]: feature for (img, feature) in zip(group, features)}
        feature_dict.update(d)

    return feature_dict

def proportion(lst, lbl_dict):

    s = set(lst)
    dic = {}
    for item in s:
        dic.update({item: lst.count(item)})

    prop = {}
    for k, v in dic.items():
        p = v / lbl_dict[k][1]
        prop[k] = p

    return max(prop, key=lambda x: prop[x])

def test(conf, threshold, model, lbl_dict, feat_all_data, mode):
    start = time.time()

    if mode == 'test':
        feat_test_data = test_data_featurize(conf.test_root, conf, model)
    elif mode == 'predict':
        feat_test_data = test_data_featurize(conf.predict_root, conf, model)

    predict = []
    for name, feat in feat_test_data.items():
        similarities = []

        for path, lst in feat_all_data.items():
            s = cosin_metric(feat, lst[0])
            info = (s, lst[-1])
            similarities.append(info)

        similarities.sort()
        t = [si[1] for si in similarities if si[0] > threshold]
        if len(t) == 0:
            # pre = similarities[0][1]
            pre = lbl_dict[similarities[0][1]][0]
        else:
            # pre = lbl_dict[max(set(p), key=t.count)][0]
            pre = lbl_dict[proportion(t, lbl_dict)][0]
            # pre = proportion(t, lbl_dict)

        predict.append(pre)

    if mode == 'test':
        label = pd.read_csv(conf.label_root)
        label = label['category'].tolist()

        label = np.array(label)
        predict = np.array(predict)

        acc = np.sum(predict == label) / len(predict)

        end = time.time()
        return acc, (end - start) / 60.0

    elif mode == 'predict':
        df = pd.DataFrame(columns=['id', 'label'])
        df['id'] = range(len(os.listdir(conf.predict_root)))
        df['label'] = predict
        df.to_csv('./' + conf.student_id + 'submission_{}.csv'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                          index=False)
        end = time.time()
        return (end - start) / 60.0

if __name__ == '__main__':

    conf = Config()
    threshold = 0.31137382984161377
    mode = 'test'

    model = FaceMobileNet(conf.embedding_size)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
    model.eval()

    lbl_dict = label_dict(conf.all_data)

    feat_all_data = all_data_featurize(conf.all_data, conf, model)

    if mode == 'test':
        acc, time = test(conf, threshold, model, lbl_dict, feat_all_data, mode)
        print("Acc: {}  Time cost: {}".format(acc, time))
    elif mode == 'predict':
        time = test(conf, threshold, model, lbl_dict, feat_all_data, mode)
        print("Time cost: {}".format(time))
