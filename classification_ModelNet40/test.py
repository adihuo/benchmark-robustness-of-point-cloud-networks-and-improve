"""
python test.py --model pointMLP --msg 20220209053148-404
"""
import argparse
import os
import shutil
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import progress_bar, IOStream
from data import ModelNet40
import sklearn.metrics as metrics
from helper import cal_loss
import numpy as np
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))
# print('model_names:', model_names)

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='20231117174824-4509', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='PointConT_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")
    if args.msg is None:
        message = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    else:
        message = "-"+args.msg
    args.checkpoint = 'checkpoints/' + args.model + message

    print('==> Preparing data..')
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Model
    print('==> Building model..')

    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])

    test_out = validate(net, test_loader, criterion, device)
    print(f"Vanilla out: {test_out}")


def validate(net, testloader, criterion, device):
    args = parse_args()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            if not args.model == 'PointConT_cls':
                data = data.permute(0, 2, 1)
            # PointCont_cls需要修改

            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }

def move_files(files, destination_directory='data/modelnet40_ply_hdf5_2048'):
    file = ['ply_data_test0.h5', 'ply_data_test1.h5']
    for file_path in file:
        file_path = os.path.join(files, file_path)
        try:
            shutil.copy(file_path, destination_directory)
        except Exception as e:
            print(f"移动文件 '{file_path}' 失败: {e}")

if __name__ == '__main__':
    for noisy in ['gaussian', 'dropout', 'rotation']:
        source_directory = os.path.join('data/noisy', noisy)
        for i in range(1, 5):
            source = source_directory
            source_directory = os.path.join(source_directory, str(i))
            # print(source_directory)
            move_files(source_directory)
            print(f'--------{noisy}-{i}----------')
            main()
            source_directory = source

    # main()
