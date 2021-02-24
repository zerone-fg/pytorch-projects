import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_roc(x, predicts, title='roc'):
    FPR, TPR, thresholds = roc_curve(x, predicts)
    roc_auc = auc(FPR, TPR)
    plt.title(title)
    plt.plot(FPR, TPR, '-', label='AUC = {:.4f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


def plot_multi_roc(labels, scores, names, corrects, title='multi roc'):
    """
    同时画出N个roc曲线
    :param labels: [[]] N个曲线的标签
    :param scores: N个scores
    :param names:  N 个曲线的名字
    :param corrects:  N个预测正确的数量
    :param title: 图像名称
    :return: None
    """
    plt.title(title)
    for label, predict, name, correct in zip(labels, scores, names, corrects):
        FPR, TPR, thresholds = roc_curve(label, predict, pos_label=correct)
        roc_auc = auc(FPR, TPR)
        plt.plot(FPR, TPR, '-', label='{}, AUC = {:.4f}'.format(name, roc_auc))
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    x = np.array([1, 1, 0, 1])
    scores1 = np.array([0.1, 0.2, 0.35, 0.8])
    scores2 = np.array([0.3, 1, 1, 0.6])
    scores3 = np.array([0.32, 0.5, 0.7, 1])
    plot_multi_roc(x, [scores1, scores2, scores3], ['r1', 'r2', 'r3'])
    plot_roc(x, scores1)
