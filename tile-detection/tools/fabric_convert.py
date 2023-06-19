# coding:utf-8

import argparse
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
    '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7, '破洞': 8, '褶子': 9,
    '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15
}


def construct_imginfo(root_dir, filename, h, w, ID):
    # template_class = template.split(".")[0]
    filename = "{}/{}/{}".format(root_dir, filename.split('.')[0], filename)
    # template = "{}/{}/template_{}".format(root_dir, filename.split("/")[1], template)
    image = {"license": 1,
             "file_name": filename,
             # 'template_name': template,
             # "cls": template_class,
             "coco_url": "xxx",
             "height": h,
             "width": w,
             "date_captured": "2019-06-25",
             "flickr_url": "xxx",
             "id": ID
             }
    return image


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


def add_normal(normal_dir, out_file):
    coco = COCO(out_file)
    ID = max(coco.getImgIds()) + 1
    annotations = mmengine.load(out_file)
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmengine.dump(annotations, out_file)


def generate_normal(normal_dir, out_file):
    cls2ind = mmengine.load("./source/cls2ind.pkl")
    ind2cls = mmengine.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ID = 0
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmengine.dump(annotations, out_file)


def generate_coco(annos, out_file):
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2023/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []

    for cat_name, cat_id in class_to_ind20.items():
        category = {"id": cat_id, "name": cat_name, "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}

    img_names = {}
    dataset_base_dir = os.path.abspath(os.path.join(out_file, "../.."))

    IMG_ID = 0
    OBJ_ID = 0
    for anno in tqdm(annos):
        name = anno['name']
        defect_name = anno["defect_name"]
        bbox = anno["bbox"]
        img_path = "{}/defect/{}/{}".format(dataset_base_dir, name.split(".")[0], name)
        if name not in img_names and os.path.exists(img_path):
            img_names[name] = IMG_ID
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            img_info = construct_imginfo("defect", name, h, w, IMG_ID)
            annotations["images"].append(img_info)
            IMG_ID = IMG_ID + 1
            img_id = img_names[name]
            cat_ID = class_to_ind20[defect_name]
            xmin, ymin, xmax, ymax = bbox
            area = (ymax - ymin) * (xmax - xmin)
            seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            ann = construct_ann(OBJ_ID, img_id, cat_ID, seg, area, bbox)
            annotations["annotations"].append(ann)

        OBJ_ID += 1
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmengine.dump(annotations, out_file)


def generate_train(coco, val):
    cls2ind = mmengine.load("./source/cls2ind.pkl")
    ind2cls = mmengine.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    anno_train = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    anno_val = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ids = coco.getImgIds()
    for imgId in ids:
        img_info = coco.loadImgs(imgId)[0]
        cls = img_info["cls"]
        ann_ids = coco.getAnnIds(img_info['id'])
        ann_info = coco.loadAnns(ann_ids)
        if cls in val:
            anno_val["images"].append(img_info)
            anno_val["annotations"] += ann_info
        else:
            anno_train["images"].append(img_info)
            anno_train["annotations"] += ann_info
    mmengine.dump(anno_train, "../data/round2_data/Annotations/anno_train.json")
    mmengine.dump(anno_val, "../data/round2_data/Annotations/anno_val.json")


def split(out_file):
    all_class = mmengine.load("./source/temp_cls.pkl")
    np.random.seed(1)
    val = np.random.choice(all_class, 28, replace=False)
    train = list(set(all_class) - set(val))
    coco = COCO(out_file)
    generate_train(coco, val)


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
    annos1 = mmengine.load(os.path.join(data_root, "Annotations/anno_train_0924.json"))
    annos2 = mmengine.load(os.path.join(data_root, "Annotations/anno_train_1004.json"))
    annos = annos1 + annos2

    out_file = "{}/Annotations/train.json".format(data_root)
    print("convert to coco format...")
    generate_coco(annos, out_file)


if __name__ == "__main__":
    main()
