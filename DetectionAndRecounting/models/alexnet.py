import torch
import torch.nn as nn
class AlexNetFeature(nn.Module):
    def __init__(self):
        # 3 * 224 * 224
        super(AlexNetFeature, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), 4, 2),
            # 96 * 55 * 55
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        # 256 * 27 * 27
        self.conv_2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), 1, 2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
        )
        # 256 * 13 * 13
        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 384, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )
        # 384 * 13 * 13
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 384, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )
        # 384 * 13 * 13
        self.conv_5 = nn.Sequential(
            nn.Conv2d(384, 256, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)
        )
        # 256 * 6 * 6

    def forward(self, x):
        x = self.conv_1(x)
        # print(x.shape)
        x = self.conv_2(x)
        # print(x.shape)
        x = self.conv_3(x)
        # print(x.shape)
        x = self.conv_4(x)
        # print(x.shape)
        x = self.conv_5(x)
        return x


if __name__ == '__main__':
    x = torch.randn([8, 3, 224, 224])
    model = AlexNetFeature()
    y = model(x)
    print(y.shape)
