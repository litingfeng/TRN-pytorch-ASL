import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN, ContrastiveLoss
from transforms import *
from opts import parser
import datasets_video

import wandb
wandb.init(project="trn")


best_prec1 = 0
best_loss  = 10

def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
    num_class = len(categories)


    args.store_name = '_'.join(['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments,
                                'bs%d' % args.batch_size])
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args.num_segments, 'RGB',
                base_model=args.arch,
                new_length=1,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                siamese=args.siamese)

    model_flow = TSN(num_class, args.num_segments, 'Flow',
                base_model=args.arch,
                new_length=2,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                siamese=args.siamese)

    crop_size = [model.crop_size, model_flow.crop_size]
    scale_size = [model.scale_size, model_flow.scale_size]
    input_mean = [model.input_mean, model_flow.input_mean]
    input_std = [model.input_std, model_flow.input_std]
    policies = [model.get_optim_policies(), model_flow.get_optim_policies()]
    train_augmentation = [model.get_augmentation(), model_flow.get_augmentation()]

    for group in policies[0]:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    for group in policies[1]:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            model.load_state_dict(base_dict)
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
            print('best prec ', best_prec1.item())
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

        if os.path.isfile(args.resume_of):
            print(("=> loading checkpoint '{}'".format(args.resume_of)))
            checkpoint = torch.load(args.resume_of)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
            model_flow.load_state_dict(base_dict)
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
            print('best prec ', best_prec1.item())
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume_of)))

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model_flow = torch.nn.DataParallel(model_flow, device_ids=args.gpus).cuda()

    wandb.watch(model)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = [GroupNormalize(input_mean[0], input_std[0]), GroupNormalize(input_mean[1], input_std[1])]
    else:
        normalize = IdentityTransform()

    # if args.modality == 'RGB':
    #     data_length = 1
    # elif args.modality in ['Flow', 'RGBDiff']:
    #     data_length = args.data_length # set 3 for rachel

    wandb.init(project="trn")
    wandb.config.update(args)

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize[0],
                   ]), siamese=args.siamese),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet('/dresden/gpu2/tl6012/data/ASL/Rachel_lexical/', args.val_list, num_segments=args.num_segments,
                   new_length=1,
                   modality='RGB',
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size[0])),
                       GroupCenterCrop(crop_size[0]),
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize[0],
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader_of = torch.utils.data.DataLoader(
        TSNDataSet('/dresden/gpu2/tl6012/data/ASL/Rachel_lexical_of/', args.val_list, num_segments=args.num_segments,
                   new_length=2,
                   modality='Flow',
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size[1])),
                       GroupCenterCrop(crop_size[1]),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize[1],
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll' and not args.siamese:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.siamese:
        criterion = torch.nn.CrossEntropyLoss().cuda() #ContrastiveLoss(margin=args.margin)
    else:
        raise ValueError("Unknown loss type")




    optimizer = torch.optim.SGD(policies[0],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        prec = validate_2stream(val_loader, val_loader_of, model, model_flow, criterion, 0)
        #validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, loss = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            if args.siamese:
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                }, is_best)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)

        # Save model to wandb
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data = [ele.cuda() for ele in data]
        if args.siamese:
            input1, input2, target = data
        else:
            input1, target = data

        # print('input1 ', input1.shape, ' ', target.shape)
        # continue

        data_time.update(time.time() - end)


        # compute output
        if args.siamese:
            output = model(input1, input2)
            print('output ',output.size(), output)
            print('target ', target.size())
            loss = criterion(1-output, target)
            prec1 = (0,)
        else:
            output = model(input1)
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target, topk=(1,))

        # measure accuracy and record loss
        losses.update(loss.item(), input1.size(0))
        top1.update(prec1[0], input1.size(0))
        #top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, lr=optimizer.param_groups[-1]['lr']))

            print(output)

            log.write(output + '\n')
            log.flush()



def validate(val_loader, model, criterion, iter, log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = [ele.cuda() for ele in data]
            if args.siamese:
                input1, input2, target = data
            else:
                input1, target = data

            # compute output
            if args.siamese:
                output1, output2 = model(input1, input2)
                loss = criterion(output1, output2, target)
                prec1 = (0,)
            else:
                output = model(input1)
                loss = criterion(output, target)
                prec1 = accuracy(output.data, target, topk=(1,))

            losses.update(loss.item(), input1.size(0))
            top1.update(prec1[0], input1.size(0))
            #top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))


                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, loss=losses))
    wandb.log({"Test Top1 Accuracy": top1.avg, "Test Loss": losses.avg})

    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    if log is not None:
        log.write(output + ' ' + output_best + '\n')
        log.flush()

    return top1.avg, losses.avg

def validate_2stream(val_loader, val_loader_of, model, model_of, criterion, iteration, log=None):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model_of.eval()

    end = time.time()
    dataiter = iter(val_loader_of)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = [ele.cuda() for ele in data]
            data_flow = next(dataiter)
            input1, target = data
            input1_flow, target_flow = data_flow
            print(i, ' input ', input1.shape, '\t', input1_flow.shape)

            output = model(input1)
            output_flow = model_of(input1_flow)

            output = torch.mean(torch.stack((output, output_flow)), axis=0)
            prec1 = accuracy(output.data, target, topk=(1,))

            top1.update(prec1[0], input1.size(0))
            #top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1))


                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    wandb.log({"Test Top1 Accuracy": top1.avg})

    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    if log is not None:
        log.write(output + ' ' + output_best + '\n')
        log.flush()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
