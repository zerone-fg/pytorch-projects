import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split

from dataset.coco_dataset import CocoDataSet
from dataset.multidataloader import MultiLoader
from dataset.vg_dataset import VGDataSet
from models.recounting import RecountingModel
from utils.utils import plot_loss


def train(args, model: RecountingModel, device, data_loader: MultiLoader, loss_fn, optimizer, epoch):
    model.train()
    length = len(data_loader)
    loss_records = [[0] for _ in data_loader.datasets]
    for batch_idx, batch in enumerate(data_loader):
        dataset_idx, batch = batch
        images, boxes, labels = batch
        images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
        output = model((images, boxes))
        output = output[args.idx]

        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_records[dataset_idx].append(loss.detach().item())
        if batch_idx % args.log_interval == 0:
            print("Training epoch {}: {}/{} : {:.2f}% of total, Loss:{:.5f}".format(
                epoch,
                batch_idx * args.batch_size,
                length,
                batch_idx * args.batch_size * 100.0 / length,
                loss.item()
            ))
        if args.single_run:
            break
    return loss_records


def test(args, model, device, test_loader: MultiLoader, loss_fn):
    model.eval()
    losses = [0.0 for _ in test_loader.datasets]
    correct = 0
    length = 0
    with torch.no_grad():
        for batch in test_loader:
            dataset_idx, batch = batch
            imgs, boxes, labels = batch
            imgs, boxes, labels = imgs.to(device), boxes.to(device), labels.to(device)
            output: torch.Tensor = model((imgs, boxes))
            output = output[args.idx]
            losses[dataset_idx] += loss_fn(output, labels).item()
            predict = output.argmax(dim=1)
            correct += predict.eq(labels.view_as(predict)).sum().item()
            length += labels.size()[0]
            if args.single_run:
                break
    print("Test Accuracy: {}/{} : {:.2f}%".format(
        correct, length,
        100.0 * correct / length
    ))
    print('Loss:')
    for i, loss in enumerate(losses):
        print('{:.5f}'.format(loss / len(test_loader.datasets[i])))


def get_loaders(args):
    if not 0 < args.train_rate <= 1:
        raise ValueError('Illegal train rate:{} !'.format(args.train_rate))
    vg_path = os.path.join(args.root_dir, 'vg')
    if args.idx == 0:
        dataset = CocoDataSet(root_dir=os.path.join(args.root_dir, 'coco'))
    elif args.idx == 1:
        dataset = VGDataSet(root_dir=vg_path, name='attr')
    elif args.idx == 2:
        dataset = VGDataSet(root_dir=vg_path, name='action')
    else:
        raise ValueError('idx must be 0, 1 or 2 !')
    total_datasets = [dataset]
    train_datasets = []
    test_datasets = []

    for dataset in total_datasets:
        train_size = int(args.train_rate * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    train_loader = MultiLoader(
        datasets=train_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=VGDataSet.collate_fn)

    test_loader = MultiLoader(
        datasets=test_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=VGDataSet.collate_fn)

    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train Recounting Model")
    parser.add_argument("--epoch", help="Training epoch", type=int, default=10)
    parser.add_argument("--root_dir", help="Dataset root directory", type=str, default="./data")
    parser.add_argument("--batch_size", help="Training batch size", type=int, default=5)
    parser.add_argument("--save_model", help="Whether to save the model", action="store_true", default=True)
    parser.add_argument("--no_cuda", help="train without cuda", action='store_true', default=False)
    parser.add_argument("--log_interval", help="Log interval for training", type=int, default=200)
    parser.add_argument("--single_run", help="Run a pass to see result", action="store_true", default=False)
    parser.add_argument("--num_workers", help="Num of workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--gamma", help="lr scheduler gamma", default=0.1)
    parser.add_argument("--step_size", help="lr step size", type=int, default=5)
    parser.add_argument("--seed", help="Random seed. Default: 5", type=int, default=1)
    parser.add_argument("--train_rate", help="split the train size and test size", type=float, default=0.8)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--pre_load", help="Load pre-trained model", action='store_true', default=False)
    parser.add_argument("--save_path", help="model path", default="./save/recount_model.pkl")
    parser.add_argument("--idx", help="0 for coco, 1 for attr, 2 for action", type=int, default=2)
    args = parser.parse_args()

    model: nn.Module = RecountingModel()
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.pre_load:
        print('Loading model')
        model = torch.load(args.save_path)
    model = model.to(device)
    torch.manual_seed(args.seed)

    print(args)
    for epoch in range(1, args.epoch + 1):
        print('\nTraining epoch {}'.format(epoch))
        data_loader, test_loader = get_loaders(args)
        train(args, model, device, data_loader, loss_fn, optimizer, epoch)
        test(args, model, device, test_loader, loss_fn)
        scheduler.step()

        if args.save_model:
            print('Saving model..')
            torch.save(model, args.save_path)
            print('Done')


if __name__ == '__main__':
    main()
