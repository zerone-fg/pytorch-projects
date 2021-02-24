import random

from torch.utils.data import Dataset, DataLoader


class MultiLoader:
    """
    整合多个数据集，但每一个batch都是从同一个数据集中取出的
    :return (idx, batch)
        idx : 对应的数据集的下标
        batch : 实际的数据
    """

    def __init__(self, datasets: list, num_workers=0, batch_size=1, drop_last=False, shuffle=False, collate_fn=None):
        self.loader_iters = []
        self.batch_size = batch_size
        self.loaders = []
        self.datasets = datasets
        for i, dataset in enumerate(self.datasets):
            data_loader = \
                DataLoader(dataset=dataset,
                           num_workers=num_workers,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           drop_last=drop_last,
                           collate_fn=collate_fn)
            self.loader_iters.append((i, iter(data_loader)))
            # self.loader_iters.append(iter(data_loader))
            self.loaders.append(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.loader_iters) == 0:
            raise StopIteration
        # idx, it = random.choice(self.loader_iters)
        idx = random.randint(0, len(self.loader_iters) - 1)
        datasetidx, it = self.loader_iters[idx]
        batch = None
        try:
            batch = it.next()
        except StopIteration:
            self.loader_iters.pop(idx)
        if batch is not None:
            return datasetidx, batch
        else:
            return self.__next__()

    def __len__(self):
        length = 0
        for loader in self.loaders:
            length += len(loader.dataset)
        return length


class TestDataSet(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    set1 = TestDataSet([0 for x in range(1, 20)])
    set2 = TestDataSet([1 for x in range(1, 10)])
    set3 = TestDataSet([2 for x in range(1, 10)])
    datasets = [set1, set2, set3]
    loader = MultiLoader(datasets=datasets, batch_size=4)
    for batch_idx, val in enumerate(loader):
        dataset_idx, batch = val
        print("Batch {} : {}, from dataset {}".format(batch_idx, batch, dataset_idx))
        # if ...
        # ...
