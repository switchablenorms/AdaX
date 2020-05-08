'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function, division

import argparse
import os
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import accuracy, AverageMeter, get_last_weights, LRScheduler, save_checkpoint
from distributed_utils import load_state
from tensorboardX import SummaryWriter
from dataloader import CIFAR10, CIFAR100
from models.ResNet import *
from models.ResNet_cifar import *
# from my_dataset import Dataset, McDataset

import yaml
from AdaX import AdaXW, AdaX

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')

# Datasets
parser.add_argument('--config', default='cfgs/cifar10/config_res18.yaml')
# parser.add_argument('--data_dir', default='data')
parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--max_steps', default=None, type=int,
                    help='number of total steps to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 128)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 100)')
parser.add_argument('--base_lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-type', default='step', type=str)
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--save_path', default='Outputs', type=str, metavar='PATH',
                    help='path to save train log, test log and ckpt (default: Outputs)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', default='ResNet20', type=str,
                    help='Network Architecture')

parser.add_argument('--depth', type=int, default=50, help='Model depth.')
parser.add_argument('--dataset', default = 'cifar10')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12)
parser.add_argument('--compressionRate', type=int, default=2)
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--save_freq', type=int, default=1500,
                    help='Frequency to save ckpt.')
parser.add_argument('--print_freq', type=int, default=100,
                    help='Frequency to print info.')
parser.add_argument('--val_freq', type=int, default=None,
                    help='Frequency to evaluate ckpt.')

parser.add_argument('--gate_sort', action='store_true', help='whether sort the gate in Autogroup')
parser.add_argument('--MCDataset', action='store_true', help='Whether use MCDataset to load data.')
parser.add_argument('--lr_multi', action='store_true',
                    help='Whether multiply the base lr by the ratio of batch_size/256.')

parser.add_argument('--block_name', default=None, type=str)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


use_cuda = True
best_acc = 0  # best test accuracy


def main():
    global args, best_acc

    with open(args.config) as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            print('###################################################')
            print('Please update pyyaml >= 5.1')
            print('###################################################')
            config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    if args.val_freq is None:
        args.val_freq = args.save_freq

    if not os.path.exists(args.save_path):
        print('Create {}.'.format(args.save_path))
        os.makedirs(args.save_path)


    print('###################################################')
    print('Parameters')
    print(args)
    print('###################################################')

    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = CIFAR10
    else:
        dataloader = CIFAR100

    train_dataset = dataloader(root=args.train_root, train=True, transform=transform_train)
    train_loader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=args.val_root, train=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    if '18' in args.arch:
        model = ResNet18()
        print('Using ResNet18 for training')
    elif '50' in args.arch:
        model = ResNet50()
        print('Using ResNet50 for training')
    else:
        model = resnet20()
        print('Using resnet20 for training')
    model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = AdaXW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # Resume
    ckpt_path = os.path.join(args.save_path, 'ckpt')
    if not os.path.exists(ckpt_path):
        print('Create {}.'.format(ckpt_path))
        os.makedirs(ckpt_path)

    weight_file = get_last_weights(ckpt_path + '/ckpt_step*.pth')
    last_step = 0
    if len(weight_file) == 0:
        print('No ckpt for resuming.')
    else:
        # Load checkpoint.
        print('==> Resuming from {}.'.format(weight_file))
        assert os.path.isfile(weight_file), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load(weight_file)
        checkpoint = load_state(weight_file, model)
        model.load_state_dict(checkpoint['state_dict'])
        if not args.evaluate:
            last_step = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_top1, test_top5 = validate(val_loader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc.top1:  %.2f, Test Acc.top5:  %.2f' % (test_loss, test_top1, test_top5))
        return

    # Tensorboard
    tb_logger = SummaryWriter(args.save_path + '/tf_log')
    if not os.path.exists(args.save_path + '/tf_log'):
        print('Create {}.'.format(args.save_path + '/tf_log'))
        os.makedirs(args.save_path + '/tf_log')

    T_max = int(len(train_dataset) * args.epochs / args.train_batch)
    try:
        T_max = max(T_max, args.max_iter)
    except:
        pass
    if args.lr_multi:
        lr_multi = float(8) * args.train_batch / (8. * 32)
    else:
        lr_multi = 1.0
    print('Totally train {} steps.'.format(T_max))
    lr_scheduler = LRScheduler(max_steps=T_max, optimizer=optimizer, lr_mult=lr_multi, args=args)
    # Train and val
    _ = train(train_loader, val_loader, model, criterion, optimizer, last_step, T_max,
              lr_scheduler, tb_logger)


def train(train_loader, val_loader, model, criterion, optimizer, last_step, max_step,
          lr_scheduler, tb_logger, use_cuda=True):
    # switch to train mode
    model.train()
    batch_time = AverageMeter(100)
    data_time = AverageMeter(100)
    forward_time = AverageMeter(100)
    backward_time = AverageMeter(100)
    losses = AverageMeter(100)
    top1 = AverageMeter(100)
    top5 = AverageMeter(100)
    end = time.time()

    curr_step = last_step
    curr_epoch = int(curr_step / len(train_loader))
    while curr_step <= max_step:
        print('Current epoch:{}'.format(curr_epoch))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            current_lr = lr_scheduler.update(curr_epoch)
            # print('t1', time.time() - end)
            # t2 = time.time()
            # measure data loading time
            data_end = time.time()
            data_time.update(data_end - end)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())

            forward_end = time.time()
            forward_time.update(forward_end - data_end)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            backward_time.update(time.time() - forward_end)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            curr_step += 1
            # print('Current step: {}'.format(curr_step))
            # Save ckpt
            if curr_step > 0 and curr_step % args.save_freq == 0:
                save_state = {
                    'step': curr_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_path = os.path.join(args.save_path, 'ckpt')
                save_checkpoint(save_state, save_path=save_path, step=curr_step)

            # Print results and save tensorboard
            if curr_step > 0 and curr_step % args.print_freq == 0:
                if tb_logger is not None:
                    tb_logger.add_scalar('Train/Loss', losses.avg, curr_step)
                    tb_logger.add_scalar('Train/Top@1', top1.avg, curr_step)
                    tb_logger.add_scalar('Train/Top@5', top5.avg, curr_step)
                    tb_logger.add_scalar('lr', current_lr, curr_step)

                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Forward_Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                      'Backward_Time {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'LR {lr:.4f}'.format(
                    curr_step, max_step, batch_time=batch_time,
                    data_time=data_time, forward_time=forward_time, backward_time=backward_time,
                    loss=losses, top1=top1, top5=top5, lr=current_lr))

            if curr_step > 0 and curr_step % args.val_freq == 0:
                val_loss, prec1, prec5 = validate(val_loader, model, criterion)
                model.train()
                if tb_logger is not None:
                    tb_logger.add_scalar('Eval/Loss', val_loss, curr_step)
                    tb_logger.add_scalar('Eval/Top@1', prec1, curr_step)
                    tb_logger.add_scalar('Eval/Top@5', prec5, curr_step)
                    tb_logger.add_scalar('Eval/Generation_Error', val_loss - losses.avg, curr_step)

                if prec1 >= best_acc:
                    save_state = {
                        'step': curr_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    save_path = os.path.join(args.save_path, 'ckpt')
                    save_checkpoint(save_state, is_best=True, save_path=save_path, step=curr_step)
            # print('t3', time.time() - end)
        print("Finish epoch:{}".format(curr_epoch))
        curr_epoch += 1

    return None


def validate(val_loader, model, criterion, use_cuda=True):
    # global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item())
        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
            # if args.regularizer == 'gp_sum':
            #     logger.info('Reg {}/{}'.format(regularizer.data, ngc))
    print(
        ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {batch_time.sum:.3f}'.format(top1=top1, top5=top5,
                                                                                          batch_time=batch_time))
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
