import torch
import torch.nn as nn
from torchvision.ops import RoIPool

from models import alexnet


class RecountingModel(nn.Module):
    def __init__(self, roi_size=(3, 3), out_features=1024, n_class=(80, 45, 25), scale=224/6.0):
        """
        :param roi_size: output size of roi pooling
        :param n_class: n_class of (object, attr, action)
        :param scale: feature_size / img size
        """
        super(RecountingModel, self).__init__()
        self.scale = scale
        self.__features = alexnet.AlexNetFeature()
        self.__roi_pooling = RoIPool(roi_size, spatial_scale=1.0)
        input_size = roi_size[0] * roi_size[1] * 256
        self.__fc = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_features),
            nn.ReLU(inplace=True)
        )
        # object 分类层
        self.__clf_object = nn.Sequential(
            nn.Linear(out_features, n_class[0]),
            # nn.Softmax(dim=1)
        )
        # attr 分类层
        self.__clf_attr = nn.Sequential(
            nn.Linear(out_features, n_class[1]),
            # nn.Softmax(dim=1)
        )
        # action 分类层
        self.__clf_action = nn.Sequential(
            nn.Linear(out_features, n_class[2]),
            # nn.Softmax(dim=1)
        )

    def features(self, x):
        """
        通过Alexnet提取整个图像的特征, 并输出每个ROI的特征
        :param x: (img, rois)
                img : tensor(C, H, W)
                rois : Tensor(K, 5)
        :return: Tensor(K, out_features)
        """
        x, rois = x
        x = self.__features(x)
        rois[:, 1:] /= self.scale
        x = self.__roi_pooling(x, rois)
        x = x.view(x.size()[0], -1)
        x = self.__fc(x)
        return x

    def forward(self, x):
        """
        :param x: (img Tensor(N, C, H, W), box Tensor(K, 5))
        :return:  object, attribute, action的分类输出
        """
        x = self.features(x)
        object_out = self.__clf_object(x)
        attr_out = self.__clf_attr(x)
        action_out = self.__clf_action(x)
        return object_out, attr_out, action_out


if __name__ == '__main__':
    model: RecountingModel = RecountingModel()
    img = torch.randn(1, 3, 224, 224)
    boxes = torch.tensor([[0, 0, 0, 5, 5], [0, 0, 0, 10, 10]], dtype=torch.float)
    input = img, boxes
    print(model(input))
    print(model.features(input))
