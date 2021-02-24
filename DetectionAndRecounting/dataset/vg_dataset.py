import json
import os
import random

import torch
import torchvision.transforms as transform
from PIL import Image

import dataset.base
from dataset.multidataloader import MultiLoader
from models.recounting import RecountingModel


class VGDataSet(dataset.base.BaseDataSet):
    """
    visual genome 数据集， 有两种类型（action ,  attr)
    """
    def __init__(self, root_dir: object, name: object = 'action', img_size: object = (224, 224)) -> object:
        self.img_size = img_size
        self.name = name
        super(VGDataSet, self).__init__(root_dir, transform=transform.Compose([
            transform.Resize(img_size),
            transform.ToTensor(),
            transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    def load_data(self):
        """
        加载vg数据集, box格式为[x1, y1, x2, y2]
        :return image_attrs [
            {
             'image_id' : 1,
             'attrs' : [["green", "tall"], ["grey"], ["parked"], ["parked"]],
             'boxes' : [[421, 500, 91, 430], [77, 791, 328, 590], [243, 295, 489, 515], [514, 537, 366, 381]]
            },
            {
             'image_id' : 2,
              ...
            },
            ...
        ]
        """
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
        """
        加载image
        :param index: 下标而非image id
        :return: PIL Image
        """
        img_data = self._data[index]
        img_name = str(img_data.get('image_id')) + '.jpg'
        img_path = os.path.join(self._root_dir, "imgs", img_name)
        img = Image.open(img_path).convert('RGB')
        return img

    def get_boxes(self, index):
        """
        加载区域
        :param index: 下标
        :return: [ [x1,y1,x2,y2], ... ]
        """
        img_data = self._data[index]
        boxes = img_data['boxes']
        h, w = self.img_size
        img = self.load_image(index)
        ori_w = img.size[0]
        ori_h = img.size[1]
        for box in boxes:
            box[0] = box[0] / ori_w * w
            box[2] = box[2] / ori_w * w
            box[1] = box[1] / ori_h * h
            box[3] = box[3] / ori_h * h
        return boxes

    def get_labels(self, index):
        """
        加载标签
        :param index:下标
        :return: Tensor[L] L个box1对应的标签
        """
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


if __name__ == '__main__':
    root = "../data/vg"
    action_dataset = VGDataSet(root_dir=root, name='action')
    attr_dataset = VGDataSet(root_dir=root, name='attr')
    datasets = [attr_dataset, action_dataset]
    loader = MultiLoader(datasets=datasets, batch_size=2, collate_fn=VGDataSet.collate_fn)
    for i, batch in enumerate(loader):
        dataset_idx, batch = batch
        imgs, boxes, labels = batch
        input = (imgs, boxes)
        model: RecountingModel = RecountingModel(roi_size=(3, 3), n_class=(80, 45,25))
        output = model(input)
        feature = model.features(input)
        print(feature.shape)
        print(labels.shape)
        print(boxes.shape)
        print(output[0].shape)
        print(output[1].shape)
        print(output[2].shape)
        break
