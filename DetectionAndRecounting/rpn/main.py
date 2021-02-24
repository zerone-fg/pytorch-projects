import torch
weights_dict = torch.load("./fasterrcnn_resnet50_fpn_coco.pth")

for k,v in weights_dict.items():
    print(k)