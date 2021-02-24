import json
import os
import random

import torch
import torchvision.transforms as transform
from PIL import Image

import dataset.base


class TestDataSet(dataset.base.BaseDataSet):
    """
    与vg数据集相似,但是没有预处理缩放操作,用来测试整个model
    """
    def __init__(self, root_dir: object, name: object = 'action', img_size: object = (224, 224)) -> object:
        self.img_size = img_size
        self.name = name
        super(TestDataSet, self).__init__(root_dir, transform=transform.Compose([
            transform.ToTensor()
        ]))

    def load_data(self):
        if not self.check_vg():
            return []
        attr_file = os.path.join(self._root_dir, '{}_data.json'.format(self.name))
        with open(attr_file) as fp:
            image_attrs = json.load(fp)
        return image_attrs

    def check_vg(self):
        required_files = [
            "action_data.json",
            "attr_data.json",
            "top_25_actions.json",
            "top_45_attrs.json"
        ]
        if not os.path.isdir(self._root_dir):
            print("DataSet root dir is not a directory!")
            return False
        paths = os.listdir(self._root_dir)
        for file in required_files:
            if file not in paths:
                print("Missing file: {} !".format(file))
                return False
        img_dir = os.path.join(self._root_dir, 'imgs')
        if not os.path.isdir(img_dir):
            print('Missing Image Directory /imgs !')
            return False
        return True

    def load_image(self, index):
        img_data = self._data[index]
        img_name = str(img_data.get('image_id')) + '.jpg'
        img_path = os.path.join(self._root_dir, "imgs", img_name)
        img = Image.open(img_path).convert('RGB')
        return img

    def get_boxes(self, index):
        img_data = self._data[index]
        boxes = img_data['boxes']
        return boxes

    def get_labels(self, index):
        img_data = self._data[index]
        total_targets = self.load_targets()
        labels = []
        attrs = img_data['attrs']
        for box_attrs in attrs:
            attr = random.choice(box_attrs)  # 随机选取一个box属性
            if attr not in total_targets:
                print('Error: attr {} not found'.format(attr))
                continue
            labels.append(total_targets.index(attr))
        out = torch.tensor(labels, dtype=torch.long).view(-1)
        return out

    def load_targets(self):
        if self.name == 'action':
            path = os.path.join(self._root_dir, 'top_25_actions.json')
        elif self.name == 'attr':
            path = os.path.join(self._root_dir, 'top_45_attrs.json')
        else:
            raise ValueError('Name must be action or attr!')
        with open(path) as fp:
            targets = json.load(fp)
        return targets

    @staticmethod
    def collate_fn(batch_data):
        """
        解决了不同数量的rois在一个batch的问题
        """
        img_out = []
        box_out = []
        label_out = []
        for i, data in enumerate(batch_data):
            img, boxes, labels = data
            img_out.append(img)
            for box in boxes:
                box.insert(0, i)
                box_out.append(box)
            for label in labels:
                label_out.append(label)
        img_out = torch.stack(img_out)
        label_out = torch.stack(label_out)
        box_out = torch.tensor(box_out, dtype=torch.float)
        return img_out, box_out, label_out.view(-1)

