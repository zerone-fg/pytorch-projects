import argparse

import torch
from torch import nn

from dataset.coco_dataset import CocoDataSet
from dataset.multidataloader import MultiLoader
from dataset.vg_dataset import VGDataSet
from models.recounting import RecountingModel
from utils.roc import plot_multi_roc


def eval_recount(model, device, test_loader, loss_fn):
    """
    评估recounting model
    """
    model.eval()
    loss = 0.0
    total_correct = 0
    length = len(test_loader)
    if length == 0:
        return
    scores = [[], [], []]
    truths = [[], [], []]
    corrects = [0, 0, 0]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            dataset_idx, batch = batch
            images, boxes, labels = batch
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
            output = model((images, boxes))
            output = output[dataset_idx]
            loss += loss_fn(output, labels).item()
            predict = output.argmax(dim=1)
            batch_correct = predict.eq(labels.view_as(predict)).sum().item()
            total_correct += batch_correct
            corrects[dataset_idx] += batch_correct
            truths[dataset_idx].extend(labels.cpu().numpy())
            for i, index in enumerate(predict.cpu().numpy()):
                scores[dataset_idx].append(output[i, index].cpu().item())
    loss /= length
    print("Test Accuracy: {}/{} : {:.2f}%, Loss:{:.5f}".format(
        total_correct, length,
        100.0 * total_correct / length,
        loss
    ))
    names = ["object", "attribute", "action"]
    plot_multi_roc(truths, scores, names, corrects, title="Recounting Model")


def get_testloader(args):
    coco_dataset = CocoDataSet(root_dir='./data/coco')
    action_dataset = VGDataSet(root_dir='./data/vg', name='action')
    attr_dataset = VGDataSet(root_dir=args.root_dir, name='attr')
    datasets = [coco_dataset, attr_dataset, action_dataset]
    loader = MultiLoader(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=VGDataSet.collate_fn)
    return loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate Recounting Model")
    parser.add_argument("--root_dir", help="Dataset root directory", type=str, default="./data/vg")
    parser.add_argument("--batch_size", help="Training batch size", type=int, default=5)
    parser.add_argument("--no_cuda", help="train without cuda", action='store_true', default=False)
    parser.add_argument("--num_workers", help="Num of workers", default=0)
    args = parser.parse_args()

    model: RecountingModel = torch.load('./save/recount_model.pkl')
    loss_fn = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    test_loader = get_testloader(args)
    model = model.to(device)
    eval_recount(model, device, test_loader, loss_fn)


if __name__ == '__main__':
    main()
