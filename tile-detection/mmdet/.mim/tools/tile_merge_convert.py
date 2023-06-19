# coding:utf-8
import argparse
import json
from tqdm import tqdm
from glob import glob
import os
from PIL import Image

categoryid2name = {
    "1": "边异常",
    "2": "角异常",
    "3": "白色点瑕疵",
    "4": "浅色块瑕疵",
    "5": "深色点块瑕疵",
    "6": "光圈瑕疵",
    "7": "记号笔",
    "8": "划伤"
}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    parser.add_argument(
        'dataset_dirs',
        type=str,
        nargs='+',
        help='path of train datasets with json format annotation')

    return parser.parse_args()


def read_json(dataset_dir: str) -> list:
    assert os.path.exists(os.path.join(dataset_dir, 'train_annos.json'))

    with open(os.path.join(dataset_dir, 'train_annos.json'), 'r', encoding='utf-8') as f:
        tile_json = json.load(f)

    assert (len(tile_json) > 0)
    return tile_json


def annotation_filter(annotation_list: list, remain_category_list: list):
    assert len(remain_category_list) > 0

    keep_categories = [cat['id'] for cat in remain_category_list]

    anno_list = [anno for anno in annotation_list if anno['category'] in keep_categories]

    category_new_mapping = {}
    for id, cat in enumerate(remain_category_list):
        cat_id = cat['id']
        category_new_mapping[cat_id] = id + 1
    print('Len: {}'.format(len(category_new_mapping.items())))

    for idx, anno_elem in enumerate(anno_list):
        cat_raw_id = anno_elem['category']
        anno_elem['category'] = category_new_mapping[cat_raw_id]

    # print(category_list)
    return anno_list


def main():
    args = parse_args()

    categories = []
    cat_id = 1
    for _, cat_name in categoryid2name.items():
        if int(_) in [3, 4, 5, 6, 8]:
            # print('_{}:_{}'.format(cat_id, cat_name))
            categories.append(dict(id=int(cat_id), name=cat_name))
            cat_id += 1

    dataset_dirs = args.dataset_dirs
    dataset_parent_dir = None
    annotation_list = []

    images = []
    image_id = 1
    anno_id = 1
    for dataset_dir in dataset_dirs:
        if dataset_parent_dir is None:
            dataset_parent_dir = os.path.abspath(os.path.join(dataset_dir, os.path.pardir))
        else:
            assert dataset_parent_dir == os.path.abspath(os.path.join(dataset_dir, os.path.pardir))

        tile_annos = read_json(dataset_dir)
        after_annos = annotation_filter(tile_annos,
                                        [dict(id=int(cat_id), name=cat_name) for cat_id, cat_name in
                                         categoryid2name.items() if int(cat_id) in [3, 4, 5, 6, 8]])

        img_name2anno = {}
        for tile_anno in after_annos:
            img_name = tile_anno['name']
            if img_name not in img_name2anno.keys():
                img_name2anno[img_name] = []
            cat_id = tile_anno['category']
            bboxes = tile_anno['bbox']
            img_name2anno[img_name].append([bboxes, cat_id])

        for image_path in tqdm(glob(os.path.join(dataset_dir, 'train_imgs/*.jpg'))):
            img_name = os.path.basename(image_path)
            if img_name in img_name2anno.keys():
                height, width = Image.open(image_path).size
                images.append({"file_name": img_name, "id": image_id, "height": height, "width": width})

                for bbox, category_id in img_name2anno[img_name]:
                    xmin, ymin, xmax, ymax = bbox
                    w = xmax - xmin
                    h = ymax - ymin
                    annotation_list.append(dict(
                        segmentation=[[]],
                        area=w * h,
                        iscrowd=0,
                        image_id=image_id,
                        bbox=bbox,
                        category_id=category_id,
                        id=anno_id
                    ))
                    anno_id += 1
                image_id += 1

    instances = {"images": images, "annotations": annotation_list, "categories": categories}
    with open(os.path.join(dataset_parent_dir, 'annotations.json'), "w", encoding='utf-8') as f:
        json.dump(instances, f, indent=1)


if __name__ == '__main__':
    main()
