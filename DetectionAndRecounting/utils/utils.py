import torch
import matplotlib.pyplot as plt


def convert_proposal(images, box_list, image_size=(224, 224)):
    h, w = image_size
    box_list = [box.clone().cpu().tolist() for box in box_list]
    result = []
    for i, data in enumerate(zip(images, box_list)):
        image, boxes = data
        ori_w = image.size()[1]
        ori_h = image.size()[2]
        for box in boxes:
            box[3] = box[3] / ori_h * h
            box[2] = box[2] / ori_w * w
            box[1] = box[1] / ori_h * h
            box[0] = box[0] / ori_w * w
            box.insert(0, i)
            result.append(box)
    return torch.tensor(result, dtype=torch.float)


def plot_loss(losses):
    plt.plot([i for i in range(0, len(losses))], losses, linewidth=3, color='b')
    plt.xlabel("x", fontsize=14, color='g')
    plt.ylabel("loss", fontsize=14, color='g')
    plt.show()
