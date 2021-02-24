import torch
from torch.utils.data import Dataset


class BaseDataSet(Dataset):
    """
    数据集
    """

    def __init__(self, root_dir, transform=None):
        self._transform = transform
        self._root_dir = root_dir
        print('Start loading data from {} ...'.format(self._root_dir))
        self._data = self.load_data()
        print('Finished.')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        boxes = self.get_boxes(index)
        labels = self.get_labels(index)
        img = self.load_image(index)
        if self._transform is not None:
            img = self._transform(img)
        return img, boxes, labels

    def load_image(self, index: int) -> object:
        pass

    def get_labels(self, index: int) -> list:
        pass

    def get_boxes(self, index: int) -> list:
        pass

    def load_data(self) -> list:
        pass



if __name__ == '__main__':
    pass
