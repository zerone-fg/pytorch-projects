import os
from typing import Any

import joblib
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM


class Detector:
    def __init__(self, name, clf: Any = OneClassSVM(kernel='rbf', nu=0.1)):
        self.name = name
        self.classifier = clf

    def fit(self, x):
        self.classifier.fit(x)
        return self

    def score(self, x):
        return self.classifier.score_samples(x)

    def predict(self, x):
        return self.classifier.predict(x)

    def load(self, path):
        filename = os.path.join(path, self.name + '.pkl')
        self.classifier = joblib.load(filename)

    def save(self, path):
        filename = os.path.join(path, self.name + '.pkl')
        joblib.dump(self.classifier, filename)


class SpecificModel:
    """
    针对不同的环境进行建模
    """
    def __init__(self, n_class=(80, 45, 25), n_features=1024):
        self.n_features = n_features
        self.abnormal_detector = Detector(name='abnormal')
        self.object_model = []
        self.attr_model = []
        self.action_model = []
        for i in range(n_class[0]):
            det = Detector(name='object_{}'.format(i))
            self.object_model.append(det)
        for i in range(n_class[1]):
            det = Detector(name='attr_{}'.format(i))
            self.attr_model.append(det)
        for i in range(n_class[2]):
            det = Detector(name='action_{}'.format(i))
            self.action_model.append(det)

    def detect(self, x):
        """
        通过整个图像的特征得到异常分数
        :param x: [n_features] 输入的图像的特征, 应该与recount model中输出的图像特征相等
        :return: 异常分数
        """
        return self.abnormal_detector.score(x)

    def recount(self, x, category, n):
        """
        :param x: float 分类分数
        :param category: str 类别 object/attr/action
        :param n: int 对应类别的具体分类
        :return: 估计的分数
        """
        if category == 'object':
            return 1/self.object_model[n].score([[x]])[0]
        elif category == 'attr':
            return 1/self.attr_model[n].score([[x]])[0]
        elif category == 'action':
            return 1/self.action_model[n].score([[x]])[0]
        else:
            raise ValueError('type must be one of {object, attr, action} !')

    def save(self, path):
        self.abnormal_detector.save(path)
        all_model = self.object_model + self.attr_model + self.action_model
        for det in all_model:
            det.save(path)

    def load(self, path):
        self.abnormal_detector.load(path)
        all_model = self.object_model + self.attr_model + self.action_model
        for det in all_model:
            det.load(path)


if __name__ == '__main__':
    SpecificModel().save('../save')
