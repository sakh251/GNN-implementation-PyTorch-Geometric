import argparse
import torch.nn as nn

import torch
from datetime import datetime
import torch.optim as optim
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from Nets import SAGENet, GNNStack
import Nets
from data_set import ppiDataset


def train(train_loader, valid_loader, task, criterion, writer, args):
    # build model
    model = Nets.__dict__[args.arch](max(train_dataset.num_node_features, 1), args.hidden_dim, train_dataset.num_classes,
                                    task=task)
    model.cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # train
    print('Training is strted ...')
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            label = label.cuda()
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % args.valid_steps == 0:
            test_acc = test(valid_loader, model, args)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)
    return model


def test(test_loader, model, args, is_validation=False):
    model.eval()
    correct = 0
    for data in test_loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.data > 0.5
            label = data.y
            label = label.cuda()
        #Hamming accuracy for multi-labeled problems
        correct += (((pred) == label).sum(dim=1)).sum()
    total = test_loader.dataset.data.num_nodes
    return correct / (args.num_classes * total)


model_names = list(('GNNStack', 'SAGENet'))
datset_names = list(('ppi', 'Planetoid'))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--path', metavar='path', default='ppi/',
                    help='path to ppi dataset')
parser.add_argument('--dataset', metavar='dataset', default='ppi',
                    help='Dataset name',
                    choices=datset_names)
parser.add_argument('-a', '--arch', metavar='ARCH', default='GNNStack',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: GNNStack)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-H', '--hidden_dim', default=50, type=int, metavar='H',
                    help='dimension of hidden embedding (default: 50)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--valid_steps', default=1, type=int, metavar='VN',
                    help='validate every N steps')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--num_classes', default=121, type=int,
                    help='number of classes')

args = parser.parse_args()

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

if args.dataset == 'ppi':
    path = args.path
    train_files = [f'{path}/train_feats.npy', f'{path}/train_labels.npy', f'{path}/train_graph_id.npy',
                   f'{path}/train_graph.json']
    valid_files = [f'{path}/valid_feats.npy', f'{path}/valid_labels.npy', f'{path}/valid_graph_id.npy',
                   f'{path}/valid_graph.json']
    test_files = [f'{path}/test_feats.npy', f'{path}/test_labels.npy', f'{path}/test_graph_id.npy',
                  f'{path}/test_graph.json']

    train_dataset = ppiDataset(path, parocced_file_name='train.proccesd', files=train_files)
    valid_dataset = ppiDataset(path, parocced_file_name='valid.proccesd', files=valid_files)
    test_dataset = ppiDataset(path, parocced_file_name='tes.proccesd', files=test_files)

    train_loader = DataLoader(train_dataset, batch_size=args.workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.workers, shuffle=False)

    # It is a multi-label dataset
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    task = 'node'

model = train(train_loader, valid_loader, task, criterion, writer, args)

test_result = test(test_loader, model, args)

print(f'test results: {test_result}')

