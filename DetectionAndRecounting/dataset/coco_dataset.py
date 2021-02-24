import json
import os

import torch
import torchvision.transforms as transform
from PIL import Image

from dataset.base import BaseDataSet


class CocoDataSet(BaseDataSet):
    def __init__(self, root_dir, img_size=(224, 224)):
        self.img_size = img_size
        self.annotation_dict = dict()
        self.categories = list()
        self.cat_map = dict()
        self.rev_map = dict()
        super(CocoDataSet, self).__init__(root_dir, transform=transform.Compose([
            transform.Resize(img_size),
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    def load_data(self):
        train_file = os.path.join(self._root_dir, 'instances_train2017.json')
        with open(train_file) as fp:
            data: dict = json.load(fp)
        # 加载注解，dict方便查找
        ann_data = data['annotations']
        for ann in ann_data:
            image_id = ann['image_id']
            if image_id not in self.annotation_dict.keys():
                self.annotation_dict[image_id] = []
            self.annotation_dict[image_id].append(ann)
        self.categories = data['categories']
        self.cat_map, self.rev_map = self.get_category_map()
        images = data['images']
        filter_images = []
        for image_data in images:
            image_id = image_data['id']
            ann = self.annotation_dict.get(image_id)
            if ann is None or len(ann) == 0:
                continue
            else:
                filter_images.append(image_data)
        return filter_images

    def load_image(self, index):
        data = self._data[index]
        image_name = data['file_name']
        image_path = os.path.join(self._root_dir, 'train2017', image_name)
        image = Image.open(image_path).convert('RGB')
        return image

    def get_labels(self, index):
        data = self._data[index]
        image_id = data['id']
        anns = self.annotation_dict.get(image_id)
        labels = []
        if anns is None:
            return torch.tensor([], dtype=torch.float)
        for ann in anns:
            labels.append(ann['category_id'])
        labels = self.ori_to_regular(labels)
        out = torch.tensor(labels, dtype=torch.long).view(-1)
        return out

    def get_boxes(self, index):
        """
        加载box
        :param index: 下标
        :return: [ [x1,y1,x2,y2], ... ]
        """
        data = self._data[index]
        image_id = data['id']
        anns = self.annotation_dict.get(image_id)
        ori_w = data['width']
        ori_h = data['height']
        h, w = self.img_size
        boxes = []
        if anns is None:
            # print('Id.{} image {} has no box!'.format(image_id, data['file_name']))
            return []
        for ann in anns:
            box = ann['bbox']
            box[2] += box[0]
            box[3] += box[1]
            box[0] = box[0] / ori_w * w
            box[2] = box[2] / ori_w * w
            box[1] = box[1] / ori_h * h
            box[3] = box[3] / ori_h * h
            boxes.append(box)
        return boxes

    def ori_to_regular(self, original_labels: list):
        regular_labels = map(lambda x: self.cat_map[x], original_labels)
        return list(regular_labels)

    def regular_to_ori(self, labels):
        original_labels = map(lambda x: self.rev_map[x], labels)
        return list(original_labels)

    def get_category_map(self):
        cates = [x['id'] for x in self.categories]
        cats_tuple = zip(cates, [i for i in range(len(cates))])
        cat_map = dict()
        rev_map = dict()
        for old_cat, new_cat in cats_tuple:
            cat_map[old_cat] = new_cat
            rev_map[new_cat] = old_cat
        return cat_map, rev_map


if __name__ == '__main__':
    root = "../data/coco"
    coco_dataset = CocoDataSet(root_dir=root)
    cats = []
    for cat in coco_dataset.categories:
        cats.append(cat['name'])
    with open("../data/coco/cats.json", 'w') as fp:
        json.dump(cats, fp)