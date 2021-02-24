import PIL.Image as Image
import cv2
import torch
import torchvision.transforms as transform
from torch import nn
from torch.optim import SGD

from dataset.multidataloader import MultiLoader
from dataset.vg_dataset import VGDataSet
from models.recounting import RecountingModel
from rpn.rpn import RPN
from rpn.train import create_model


def get_loader(root_dir='../data/vg'):
    dum_dataset = VGDataSet(root_dir=root_dir, name='action')
    action_dataset = VGDataSet(root_dir=root_dir, name='action')
    attr_dataset = VGDataSet(root_dir=root_dir, name='attr')
    datasets = [dum_dataset, attr_dataset, action_dataset]
    loader = MultiLoader(
        datasets=datasets,
        batch_size=5,
        collate_fn=VGDataSet.collate_fn)
    return loader


def debug_model_parameter():
    model: RecountingModel = RecountingModel()
    loader = get_loader()
    optimizer = SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(1)
    params = list(model.named_parameters())
    ori = [x[1].data.clone().detach() for x in params]
    for i, batch in enumerate(loader):
        dataset_idx, batch = batch
        images, boxes, labels = batch
        output = model((images, boxes))
        output = output[dataset_idx]

        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        print('batch {}: parameter change:'.format(i))
        parameters = list(model.named_parameters())

        for origin, param in zip(ori, parameters):
            print(param[0])
            param = param[1].data
            print(origin.eq(param))
        if i == 10:
            break


def show_image(boxes, image_path):
    origion_image = cv2.imread(image_path)
    for box in boxes:
        x1, y1, x2, y2 = tuple(box)
        cv2.rectangle(origion_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("result", origion_image)
    cv2.waitKey(0)


def rpn_test():
    model: RPN = create_model(6)
    weights_dict = torch.load("../save/rpn.pth")
    model_dict = model.state_dict()
    for k, v in weights_dict.items():
        if k in model_dict.keys():
            model_dict[k] = v
    model.load_state_dict(model_dict)
    model.eval()
    # path = 'E:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\\2007_000027.jpg'
    path = '../data/coco/train2017/000000000009.jpg'
    image = Image.open(path)
    image = transform.ToTensor()(image)
    boxes = model([image], None)
    print(image.shape)
    for box in boxes:
        show_image(box, path)
        print(box)


if __name__ == '__main__':
    rpn_test()