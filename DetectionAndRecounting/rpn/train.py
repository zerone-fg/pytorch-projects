import torch
import rpn.transforms
import torch.utils as utils
from rpn.my_dataset import VOC2012DataSet
from rpn.resnet50_fpn_model import resnet50_fpn_backbone
from rpn import rpn, transforms
import rpn.train_eval_utils as utils
import cv2


def visual_proposal(proposals, image_path):
    origion_image = cv2.imread(image_path)
    props_list = proposals[0]
    props_list = props_list.cpu().numpy().tolist()
    for index in range(len(props_list)):
        x1 = int(props_list[index][0])
        y1 = int(props_list[index][1])
        x2 = int(props_list[index][2])
        y2 = int(props_list[index][3])
        cv2.rectangle(origion_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("result", origion_image)
    cv2.waitKey(0)


def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = rpn.RPN(backbone=backbone, num_classes=6)
    return model


def main():
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = "H:/VOCtrainval_11-May-2012/"
    # load train data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0, collate_fn=utils.collate_fn)

    # load validation data set
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0, collate_fn=utils.collate_fn)
    model = create_model(6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_dict = torch.load("./fasterrcnn_resnet50_fpn_coco.pth")
    model_dict = model.state_dict()
    for k, v in weights_dict.items():
        if k in model_dict.keys():
            model_dict[k] = v
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    for image, targets, image_path in val_data_set_loader:
        image_path = image_path[0]
        print(image_path)
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        proposals = model(image, targets)
        visual_proposal(proposals, image_path)


if __name__ == '__main__':
    main()