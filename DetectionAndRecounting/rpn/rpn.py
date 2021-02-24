from torch import nn

from rpn.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from rpn.transform import GeneralizedRCNNTransform


class RPN(nn.Module):
    def __init__(self, backbone, num_classes=None, training=True):
        super(RPN, self).__init__()
        self.training = training
        self.backbone = backbone
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 100  # rpn中在nms处理前保留的proposal数(根据score)
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 10  # rpn中在nms处理后保留的proposal数
        rpn_nms_thresh = 0.7  # rpn中进行nms处理时使用的iou阈值
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3  # rpn计算损失时，采集正负样本设置的阈值
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
        out_channels = backbone.out_channels
        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        min_size = 480
        max_size = 640
        rpn_anchor_generator = AnchorsGenerator(
            anchor_sizes, aspect_ratios
        )
        # 生成RPN通过滑动窗口预测网络部分
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        # 定义整个RPN框架
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        # 对数据进行标准化，缩放，打包成batch等处理部分
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets):
        images, targets = self.transform(images, targets)  # 对图像进行预处理
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        proposals, proposal_losses = self.rpn(images, features, targets)
        return proposals
