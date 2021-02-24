import json
import os

import torch
from torch import nn

from models.detector import SpecificModel
from models.recounting import RecountingModel
from rpn.train import create_model
from utils.utils import convert_proposal
import torchvision.transforms as transform


class Model:
    """
    总模型， 需要事先训练好各个阶段的模型
    预训练的模型保存在root_dir 目录下
    """

    def __init__(self, roo_dir=None, n_class=(80, 45, 25), features=1024):
        self.roo_dir = roo_dir
        cats = os.path.join('./data', 'coco', 'cats.json')
        with open(cats) as fp:
            self.objects: list = json.load(fp)
        action_path = os.path.join('./data', 'vg', 'top_25_actions.json')
        with open(action_path) as fp:
            self.actions: list = json.load(fp)
        attr_path = os.path.join('./data', 'vg', 'top_45_attrs.json')
        with open(attr_path) as fp:
            self.attrs: list = json.load(fp)

        self.transform = transform.Compose([
            transform.ToPILImage(),
            transform.Resize(224),
            transform.ToTensor(),
            transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.extractor = None
        self.recounting_model: RecountingModel = RecountingModel(n_class=n_class, out_features=features)
        self.specific_model = SpecificModel(n_class=n_class, n_features=features)
        # recounting model
        filename = os.path.join(self.roo_dir, 'recount_model.pkl')
        self.recounting_model = torch.load(filename)
        # detector
        self.specific_model.load(self.roo_dir)
        # 加载rpn网络
        self.extractor: nn.Module = create_model(6)
        rpn_file = os.path.join(self.roo_dir, 'rpn.pth')
        weights_dict = torch.load(rpn_file)
        model_dict = self.extractor.state_dict()
        for k, v in weights_dict.items():
            if k in model_dict.keys():
                model_dict[k] = v
        self.extractor.load_state_dict(model_dict)

    def detect(self, img):
        """
        :param img: Tensor(N, C, H, W)
        :return: List[N] 异常分数
        """
        self.extractor.eval()
        boxes = self.extractor(img, None)
        rois = convert_proposal(img, boxes).to("cuda")
        img = self.transform(img.clone().cpu().squeeze(0)).unsqueeze(0).to("cuda")
        input = img, rois
        feature = self.recounting_model.features(input)
        return self.specific_model.detect(feature.clone().cpu().numpy())

    def recount(self, x):
        """
        :param x: Tensor(1, C, H, W) 输入的图像
        :return: (object_scores[N], attr_scores[N], action_scores[N])
        """
        with torch.no_grad():
            self.extractor.eval()
            objects = []
            attrs = []
            actions = []
            boxes = self.extractor(x, None)
            rois = convert_proposal(x, boxes).to("cuda")
            x = self.transform(x.clone().cpu().squeeze(0)).unsqueeze(0).to("cuda")
            input = x, rois
            object_out, attr_out, action_out = self.recounting_model(input)
            object_out = torch.max(object_out, 1)
            attr_out = torch.max(attr_out, 1)
            action_out = torch.max(action_out, 1)
            for idx in range(rois.size()[0]):
                object_score, object_index = object_out[0][idx].cpu().numpy(), object_out[1][idx].cpu().numpy()
                attr_score, attr_index = attr_out[0][idx].cpu().numpy(), attr_out[1][idx].cpu().numpy()
                action_score, action_index = action_out[0][idx].cpu().numpy(), action_out[1][idx].cpu().numpy()
                objects.append(
                    (self.specific_model.recount(object_score, 'object', object_index)
                     , self.objects[object_index])
                )
                attrs.append(
                    (self.specific_model.recount(attr_score, 'attr', attr_index)
                     , self.attrs[attr_index])
                )
                actions.append(
                    (self.specific_model.recount(action_score, 'action', action_index)
                     , self.actions[action_index])
                )
        return boxes, objects, attrs, actions



if __name__ == '__main__':
    model = Model()
