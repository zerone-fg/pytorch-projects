import torch
from PIL import ImageDraw
from torch.utils.data import DataLoader
import matplotlib


import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from dataset.testdataset import TestDataSet
from dataset.vg_dataset import VGDataSet
from models.model import Model

matplotlib.use('TkAgg')


def tensor_to_PIL(tensor: torch.Tensor):
    """
    :param tensor: Tensor[N, C, H, W]
    :return: images [N]
    """
    images = tensor.cpu().clone()
    image_list = []
    for img in images:
        pil_img = transforms.ToPILImage()(img)
        image_list.append(pil_img)
    return image_list


def evaluate(model: Model, device, loader):
    model.extractor.to(device)
    model.recounting_model.to(device)
    with torch.no_grad():
        for data in loader:
            image_batch, boxes, labels = data
            image_batch, boxes, labels = image_batch.to(device), boxes.to(device), labels.to(device)
            scores = model.detect(image_batch)
            print(scores)
            rois, objects, attrs, actions = model.recount(image_batch)
            boxes = []
            for box_tensor in rois:
                boxes.extend(box_tensor.tolist())

            # images = tensor_to_PIL(image_batch)  # 打开这个以
            # draw = ImageDraw.Draw(images[0])     # 一次查看一个图片
            for i, item in enumerate(zip(boxes, objects, attrs, actions)):
                images = tensor_to_PIL(image_batch)  # 打开这个以
                draw = ImageDraw.Draw(images[0])     # 查看每个proposal
                proposal = item[0]
                # print('Proposal {}\n object score: {}, attr score: {}, action score: {}'
                #       .format(i+1, item[1][0], item[2][0], item[3][0]))
                # print('object: {}, attr: {}, action: {}'.format(item[1][1], item[2][1], item[3][1]))
                draw.rectangle(tuple(proposal), outline='red', width=4)
                draw.text((proposal[0], proposal[1]), str(i+1))

                plt.subplot(121)
                plt.imshow(images[0])
                plt.subplot(122)
                plt.title('abnormal scores')
                y = [item[1][0], item[2][0], item[3][0]]
                plt.bar(range(3), y, align='center', color='steelblue')
                plt.xticks(range(3), [item[1][1], item[2][1], item[3][1]])
                plt.ylabel('score')
                plt.show() # 打开这个来逐个proposal 查看
            # plt.imshow(images[0])
            # plt.show()  # 打开这个每一张图片看一次


def main():
    model = Model(roo_dir='./save')
    device = "cuda"
    action_dataset = TestDataSet(root_dir='./data/vg', name='action')
    loader = DataLoader(dataset=action_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=VGDataSet.collate_fn
                        )
    evaluate(model, device, loader)


if __name__ == '__main__':
    main()
