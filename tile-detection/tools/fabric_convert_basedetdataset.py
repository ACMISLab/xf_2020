# coding:utf-8

import argparse
import logging
import os
import cv2
import shutil
import numpy as np
import mmengine

from tqdm import tqdm
from glob import glob

from pycocotools.coco import COCO
from icecream import ic
from collections import Counter

class_to_ind20 = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15,
    '星跳': 16, '跳花': 16, '断氨纶': 17,
    '稀密档': 18, '浪纹档': 18, '色差档': 18,
    '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19,
    '死皱': 20, '云织': 20, '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20
}

classes = ('Background',
           '破洞', '污渍', '三丝', '结头', '花板跳', '百脚', '毛粒',
           '粗经', '松经', '断经', '吊经', '粗维', '纬缩', '浆斑', '整经结',
           '跳花', '断氨纶',
           '档缺陷',
           '痕缺陷',
           '经纬'
           )


def construct_ann(obj_id, ID, category_id, seg, area, bbox):
    ann = {"id": obj_id,
           "image_id": ID,
           "category_id": category_id,
           "segmentation": seg,
           "area": area,
           "bbox": bbox,
           "iscrowd": 0,
           }
    return ann


def split(data_list, train_val_size=0.85):
    np.random.seed(2333)
    val_size = int(len(data_list) * (1.0 - train_val_size))
    rnd_indices = np.random.choice(len(data_list), size=val_size)
    # val_set = data_list[rnd_indices]

    val_set = [data_list[i] for i in rnd_indices]

    train_indices = list(set(np.arange(len(data_list))) - set(rnd_indices))
    train_set = [data_list[j] for j in train_indices]

    return train_set, val_set


def generate_COCODataset(annos, out_file):
    categories = []

    for cat_id, cat_name in enumerate(classes):
        category = {"id": cat_id, "name": cat_name}
        categories.append(category)

    annotations = {"images": [], "annotations": [], "categories": categories}
    img_names = {}
    dataset_base_dir = os.path.abspath(os.path.join(out_file, "../.."))

    IMG_ID = 0
    OBJ_ID = 0
    for anno in tqdm(annos):
        name = anno['name']
        defect_name = anno["defect_name"]
        xmin, ymin, xmax, ymax = anno["bbox"]
        if (xmin >= xmax) or (ymin >= ymax):
            logging.info('Skip: {}'.format(anno))
            continue

        cat_id = class_to_ind20[defect_name]
        area = (ymax - ymin) * (xmax - xmin)
        seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

        img_path = "{}/defect_Images/{}".format(dataset_base_dir, name)
        if name not in img_names:
            img_names[name] = IMG_ID
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                annotations['images'].append(dict(
                    file_name=name,
                    height=h,
                    width=w,
                    id=IMG_ID,
                ))
                ann = construct_ann(OBJ_ID, IMG_ID, cat_id, seg, area, bbox)
                annotations["annotations"].append(ann)

                IMG_ID += 1
            else:
                print('{} not exists'.format(img_path))
        else:
            img_id = img_names[name]
            ann = construct_ann(OBJ_ID, img_id, cat_id, seg, area, bbox)
            annotations["annotations"].append(ann)

        OBJ_ID += 1
    print(len(annotations["images"]))
    mmengine.dump(annotations, out_file)


def generate_BaseDetDataset(annos, out_file):
    categories = []

    img_names = {}
    dataset_base_dir = os.path.abspath(os.path.join(out_file, "../.."))

    IMG_ID = 0
    count = 0
    current = ''
    for anno in tqdm(annos):
        name = anno['name']
        defect_name = anno["defect_name"]
        img_path = "{}/defect_Images/{}".format(dataset_base_dir, name)
        x_min, y_min, x_max, y_max = anno["bbox"]

        # if x_min >= x_max or y_min >= y_max:
        #     print('Skip: {}'.format(name))
        #     continue

        if name not in img_names:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                assert x_min < x_max, '{}: X_min > X_max, {}'.format(name, anno)
                assert y_min < y_max, '{}: Y_min > Y_max, {}'.format(name, anno)

                img_names[name] = dict(
                    img_path="{}/{}".format(name.split(".")[0], name),
                    height=h,
                    width=w,
                    img_id=IMG_ID,
                    instances=[dict(
                        bbox=[x_min, y_min, x_max, y_max],
                        bbox_label=class_to_ind20[defect_name],
                        ignore_flag=0
                    )]
                )
                IMG_ID = IMG_ID + 1
            else:
                logging.info('{} not exists'.format(name))
        else:
            img_names[name]['instances'].append(dict(
                bbox=anno['bbox'],
                bbox_label=class_to_ind20[defect_name],
                ignore_flag=0
            ))

    data_list = list(img_names.values())
    print(len(data_list))

    # a = open(out_file, 'w')
    # a.close()

    metainfo = dict(
        classes=classes
    )
    train_list, val_list = split(data_list)

    train_dataset = dict(metainfo=metainfo, data_list=train_list)
    val_dataset = dict(metainfo=metainfo, data_list=val_list)

    # with open(os.path.join(os.path.dirname(out_file), 'train_dataset.json'), 'w') as f:
    mmengine.dump(train_dataset, os.path.join(os.path.dirname(out_file), 'train_dataset.json'))
    mmengine.dump(val_dataset, os.path.join(os.path.dirname(out_file), 'val_dataset.json'))


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='path of train datasets with json format annotation')

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = args.dataset_dir

    annos = mmengine.load(os.path.join(data_root, "Annotations/anno_train.json"))

    out_file = "{}/Annotations/train.json".format(data_root)
    print("convert to BaseDetDataset format...")
    # generate_BaseDetDataset(annos, out_file)
    generate_COCODataset(annos, out_file)


if __name__ == "__main__":
    main()
