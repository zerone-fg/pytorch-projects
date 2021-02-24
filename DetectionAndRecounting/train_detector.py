import json
import os

import torch
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, random_split

from dataset.multidataloader import MultiLoader
from dataset.vg_dataset import VGDataSet
from models.detector import Detector
from models.recounting import RecountingModel
from rpn.train import create_model
from utils.utils import convert_proposal


def get_scores(loader: MultiLoader, model: RecountingModel, extractor, n_class=(80, 45, 25), device="cuda"):
    img_features = []
    objects = [[] for _ in range(n_class[0])]
    attrs = [[] for _ in range(n_class[1])]
    actions = [[] for _ in range(n_class[2])]
    extractor.eval()
    with torch.no_grad():
        for data in loader:
            image_batch, boxes, labels = data
            image_batch, boxes, labels = image_batch.to(device), boxes.to(device), labels.to(device)
            rois = extractor(image_batch, None)
            # rois = convert_proposal(image_batch, rois).to(device)
            input = image_batch, rois
            batch_features = model.features(input).cpu().numpy()
            object_out, attr_out, action_out = model(input)
            object_out, attr_out, action_out = object_out.cpu().numpy(), attr_out.cpu().numpy(), action_out.cpu().numpy()
            for i in range(n_class[0]):
                objects[i].extend(object_out[:, i].reshape(-1, 1))
            for i in range(n_class[1]):
                attrs[i].extend(attr_out[:, i].reshape(-1, 1))
            for i in range(n_class[2]):
                actions[i].extend(action_out[:, i].reshape(-1, 1))
            img_features.extend(batch_features)
    return img_features, objects, attrs, actions


def train(data_loader, model, roi_extractor, path='./save', n_class=(80, 45, 25)):
    abnormal_detector = Detector(name='abnormal')
    object_model = [Detector(name='object_{}'.format(i)) for i in range(n_class[0])]
    attr_model = [Detector(name='attr_{}'.format(i)) for i in range(n_class[1])]
    action_model = [Detector(name='action_{}'.format(i)) for i in range(n_class[2])]
    print('Calc scores..')
    img_features, objects, attrs, actions = get_scores(data_loader, model, roi_extractor, n_class)
    print('Fit detector..')
    abnormal_detector.fit(img_features).save(path)
    for i in range(n_class[0]):
        print('Fit model: {}'.format(object_model[i].name))
        object_model[i].fit(objects[i]).save(path)
    for i in range(n_class[1]):
        print('Fit model: {}'.format(attr_model[i].name))
        attr_model[i].fit(attrs[i]).save(path)
    for i in range(n_class[2]):
        print('Fit model: {}'.format(action_model[i].name))
        action_model[i].fit(actions[i]).save(path)


def get_loader(**kwargs):
    """
    使用不同的dataset进行训练完成对特定环境的建模
    """
    root_dir = kwargs['root_dir']
    rate = kwargs['train_rate']
    # coco_dataset = CocoDataSet(root_dir=os.path.join(root_dir, 'coco'))
    vg_path = os.path.join(root_dir, 'vg')
    action_dataset = VGDataSet(root_dir=vg_path, name='action')
    # attr_dataset = VGDataSet(root_dir=vg_path, name='attr')

    dataset = action_dataset
    train_size = int(rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, test_size])

    loader = DataLoader(dataset=train_dataset,
                        batch_size=10,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=VGDataSet.collate_fn
                        )
    return loader


def get_rpn_model():
    model = create_model(6)
    weights_dict = torch.load("./save/rpn.pth")
    model_dict = model.state_dict()
    for k, v in weights_dict.items():
        if k in model_dict.keys():
            model_dict[k] = v
    return model


if __name__ == '__main__':
    args = {
        'root_dir': './data',
        'train_rate': 0.1
    }
    loader = get_loader(**args)
    recount_model = torch.load("./save/recount_model.pkl")
    extractor = get_rpn_model()
    recount_model.to("cuda")
    extractor.to("cuda")
    train(loader, recount_model, extractor)
