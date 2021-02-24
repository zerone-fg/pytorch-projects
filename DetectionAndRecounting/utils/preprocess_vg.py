import json
import os
from collections import OrderedDict

import cv2


class Preprocessor:
    def __init__(self, root_dir):
        self.__root_dir = root_dir

    def simplify_attributes(self):
        """
        将vg自带的attributes.json精简
        在相同的目录下生成 simple_attrs.json:
            [
                {
                    'image_id' : 1,
                    'attrs': [['green', 'tall'], ['red', ...] ],
                    'boxes': [[x1, y1, x2, y2], [...], ... ]
                 },
                 {
                    'image_id': 2,
                    ...
                 },
                 ...
            ]
        每一个attrs对应一个box的属性
        :param self: 数据集根目录
        """
        attr_file = os.path.join(self.__root_dir, 'attributes.json')
        image_attrs = list()
        with open(attr_file) as fp:
            data = json.load(fp)
            for img_data in data:  # for every image
                simple_data = dict()
                simple_data['image_id'] = img_data['image_id']
                simple_data['boxes'] = list()
                simple_data['attrs'] = list()
                attr_list = img_data.get('attributes')
                if attr_list is None:
                    continue
                for attr_dict in attr_list:  # for every box
                    attrs = attr_dict.get('attributes')
                    if attrs is None or len(attrs) == 0:
                        continue
                    simple_data['attrs'].append(attrs)
                    simple_data['boxes'].append(self.convert_box(attr_dict))
                image_attrs.append(simple_data)
        save_path = os.path.join(self.__root_dir, 'simple_attrs.json')
        with open(save_path, 'w') as fp:
            json.dump(image_attrs, fp)

    def get_all_attrs(self):
        paths = os.listdir(self.__root_dir)
        if 'simple_attrs.json' not in paths:
            self.simplify_attributes()
        all_attrs = dict()
        with open(os.path.join(self.__root_dir, 'simple_attrs.json')) as fp:
            data = json.load(fp)
            for img_data in data:
                for attr_list in img_data['attrs']:
                    for attr in attr_list:
                        attr = attr.strip()
                        if all_attrs.get(attr) is None:
                            all_attrs[attr] = 1
                        else:
                            all_attrs[attr] = all_attrs[attr] + 1
        return all_attrs

    def show_img_by_attr(self, attr_str):
        img_id = -1
        with open(os.path.join(self.__root_dir, 'simple_attrs.json')) as fp:
            data = json.load(fp)
            for img_data in data:
                for attr_list in img_data['attrs']:
                    for attr in attr_list:
                        attr = attr.strip()
                        if attr == attr_str:
                            img_id = img_data['image_id']
        if img_id < 0:
            print("noe found {}".format(attr_str))
        else:
            img_name = str(img_id) + '.jpg'
            img_path = os.path.join(self.__root_dir, "imgs", img_name)
            img = cv2.imread(img_path)
            cv2.imshow("img", img)
            cv2.waitKey(0)

    def convert_box(self, a: dict):
        x1 = a.get('x')
        y1 = a.get('y')
        x2 = a.get('w') + x1
        y2 = a.get('h') + y1
        return [x1, y1, x2, y2]

    def get_top_attrs(self, n):
        all = self.get_all_attrs()
        sorted_dict = OrderedDict(sorted(all.items(), key=lambda x: x[1], reverse=True))
        attrs = []
        for i, item in enumerate(sorted_dict.items()):
            key, val = item
            if str(key).endswith("ing"):
                continue
            attrs.append(key)
            n = n - 1
            if n <= 0:
                break
        return attrs

    def get_top_action(self, n):
        all = self.get_all_attrs()
        sorted_dict = OrderedDict(sorted(all.items(), key=lambda x: x[1], reverse=True))
        attrs = []
        for i, item in enumerate(sorted_dict.items()):
            key, val = item
            if str(key).endswith("ing"):
                n = n - 1
                attrs.append(key)
            if n <= 0:
                break
        return attrs

    def generate_data(self, type='action', action_num=25, attr_num=45):
        """
        :param action_num: 出现次数最多的action的数量
        :param attr_num: 同上
        :param type: 'action' or 'attr' , 生成对应的数据集
        :return:
        """
        paths = os.listdir(self.__root_dir)
        if 'simple_attrs.json' not in paths:
            self.simplify_attributes()
        if type == 'action':
            value_set = set(self.get_top_action(action_num))
        elif type == 'attr':
            value_set = set(self.get_top_attrs(attr_num))
        else:
            raise ValueError("type must be action or attr")
        filtered_data = list()
        with open(os.path.join(self.__root_dir, 'simple_attrs.json')) as fp:
            data = json.load(fp)
            for idx, item in enumerate(data):
                attrs = item['attrs']
                boxes = item['boxes']
                new_attrs = []
                new_boxes = []
                if not len(attrs) == len(boxes):
                    continue
                for box_attrs, box in zip(attrs, boxes):
                    new_box_attrs = []
                    for attr in box_attrs:
                        if attr in value_set:
                            new_box_attrs.append(attr)
                    if len(new_box_attrs) == 0:
                        continue
                    new_attrs.append(new_box_attrs)
                    new_boxes.append(box)
                if len(new_attrs) == 0:
                    continue
                item['attrs'] = new_attrs
                item['boxes'] = new_boxes
                filtered_data.append(item)
        with open(os.path.join(self.__root_dir, '{}_data.json'.format(type)), 'w') as fp:
            json.dump(filtered_data, fp)

    def run(self):
        print('generate simplified file...')
        self.simplify_attributes()
        print('generate action file...')
        self.generate_data(type='action', action_num=25)
        print('generate attr file..')
        self.generate_data(type='attr', attr_num=45)
        print('Done.')


DEFAULT_ROOT = "../data/vg"


if __name__ == '__main__':
    m = Preprocessor(DEFAULT_ROOT)
    m.run()
