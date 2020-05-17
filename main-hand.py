'''
Use hand boundbing boxes to train TRN, predict sign label and handshape simultanuously.
'''

import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset_joint import TSNDataSet
from models import TSN, ContrastiveLoss, HandTSN
from transforms import *
from opts import parser
import datasets_video
from class_balanced_loss import CB_loss

import wandb
wandb.init(project="trn-dai")


best_prec1 = 0
best_loss  = 10
HAND_CLASS = 85

def initialize_model(num_class, args):
    model = HandTSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                new_length=args.data_length,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                siamese=args.siamese
                )

    crop_size = model.model_hand.crop_size
    scale_size = model.model_hand.scale_size
    input_mean = model.model_hand.input_mean
    input_std = model.model_hand.input_std
    policies = model.model_hand.get_optim_policies()
    train_augmentation = model.model_hand.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    return model,crop_size, scale_size, input_mean, \
            input_std, policies, train_augmentation

def initialize_dataloader(args, crop_size, scale_size, input_mean,
                            input_std, train_augmentation, prefix):
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = args.data_length # set 3 for rachel

    trainset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                          new_length=data_length,
                          modality=args.modality,
                          image_tmpl=prefix,
                          transform=torchvision.transforms.Compose([
                              train_augmentation,
                              Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                              ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                              normalize,
                          ]), siamese=args.siamese, hand=args.hand)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), hand=args.hand),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()
    args.hand = True

    #os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
    num_class = len(categories)

    args.store_name = '_'.join(['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments,
                                'newL%d'% args.data_length, 'loss%s'% args.loss_type, 'bs%d' % args.batch_size, 'lambHand%.1f'% args.lamdb_hand,
                                'lr%f' % args.lr, 'onlyhand'])
    print('storing name: ' + args.store_name)

    model,crop_size, scale_size, input_mean, \
    input_std, policies, train_augmentation = initialize_model(num_class, args)


    wandb.watch(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader = initialize_dataloader(args, crop_size, scale_size, input_mean,
                                                    input_std, train_augmentation, prefix)

    #wandb.init(project="trn")
    wandb.config.update(args)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll' and not args.siamese:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'CB':
        criterion = CB_loss
    elif args.siamese:
        criterion = torch.nn.CrossEntropyLoss().cuda() #ContrastiveLoss(margin=args.margin)
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
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
        #torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sign = AverageMeter()
    losses_hand = AverageMeter()
    top1 = AverageMeter()
    top1_hand = AverageMeter()
    #top5 = AverageMeter()

    if args.no_partialbn:
        model.module.model_hand.partialBN(False)
    else:
        model.module.model_hand.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data = [ele.cuda() for ele in data]

        input, target, input_hand, target_hand = data # rgb(bs, #seg*3, 224,224), flow(bs, #seg*datalength*2, 224,224), 2 for flow_x&flow_y
        input_hand = input_hand.view(-1, 6, input_hand.size(2), input_hand.size(3)) # each sample is a pair of start&end hand images
        target_hand = target_hand.view(-1)

        data_time.update(time.time() - end)

        # compute output
        output, output_hand = model(input_hand)
        output_hand = output_hand.view(output_hand.shape[0]*2, -1)

        loss_sign = criterion(output, target)
        loss_hand = criterion(output_hand, target_hand)
        loss = loss_sign + loss_hand * args.lamdb_hand

        prec1 = accuracy(output.data, target, topk=(1,))
        prec1_hand = accuracy(output_hand.data, target_hand, topk=(1,))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        losses_sign.update(loss_sign.item(), input.size(0))
        losses_hand.update(loss_hand.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top1_hand.update(prec1_hand[0], input.size(0))
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
                    'Loss_sign{loss_sign.val:.4f} ({loss_sign.avg:.4f})\t'
                    'Loss_hand {loss_hand.val:.4f} ({loss_hand.avg:.4f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prechand@1 {top1_hand.val:.3f} ({top1_hand.avg:.3f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, loss_sign=losses_sign,
                        loss_hand=losses_hand, top1_hand=top1_hand,
                        top1=top1, lr=optimizer.param_groups[-1]['lr']))

            print(output)

            log.write(output + '\n')
            log.flush()

    wandb.log({"Train Top1 Accuracy": top1.avg, "Train Loss": losses.avg,
               "Train hand Top1 Accuracy": top1_hand.avg, "Train hand Loss": losses_hand.avg,
               "Train sign Loss": losses_sign.avg})



def validate(val_loader, model, criterion, iter, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_sign = AverageMeter()
    losses_hand = AverageMeter()
    top1 = AverageMeter()
    top1_hand = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = [ele.cuda() for ele in data]

            input, target, input_hand, target_hand = data  # rgb(bs, #seg*3, 224,224), flow(bs, #seg*datalength*2, 224,224), 2 for flow_x&flow_y
            input_hand = input_hand.view(-1, 6, input_hand.size(2),
                                         input_hand.size(3))  # each sample is a pair of start&end hand images
            target_hand = target_hand.view(-1)

            #print('input1 ', input1.shape, ' ', target.shape)
            # exit()
            # # continue

            # compute output
            output, output_hand = model(input_hand)
            # print('out ', output.shape)

            output_hand = output_hand.view(output_hand.shape[0] * 2, -1)
            loss_sign = criterion(output, target)
            loss_hand = criterion(output_hand, target_hand)
            loss = loss_sign + loss_hand * args.lamdb_hand

            prec1 = accuracy(output.data, target, topk=(1,))
            prec1_hand = accuracy(output_hand.data, target_hand, topk=(1,))


            losses.update(loss.item(), input.size(0))
            losses_sign.update(loss_sign.item(), input.size(0))
            losses_hand.update(loss_hand.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top1_hand.update(prec1_hand[0], input.size(0))
            #top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss_sign{loss_sign.val:.4f} ({loss_sign.avg:.4f})\t'
                        'Loss_hand {loss_hand.val:.4f} ({loss_hand.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prechand@1 {top1_hand.val:.3f} ({top1_hand.avg:.3f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, loss_sign=losses_sign,
                        loss_hand=losses_hand, top1_hand=top1_hand,
                        top1=top1))


                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prechand@1 {top1_hand.avg:.3f} '
              'Loss {loss.avg:.5f} '
              'Loss_hand {loss_hand.avg:.5f} '
              'Loss_sign {loss_sign.avg:.5f} '
          .format(top1=top1, top1_hand=top1_hand,loss=losses, loss_sign=losses_sign, loss_hand=losses_hand))
    wandb.log({"Test Top1 Accuracy": top1.avg, "Test Loss": losses.avg,
               "Test hand Top1 Accuracy": top1_hand.avg, "Test hand Loss": losses_hand.avg,
               "Test sign Loss": losses_sign.avg})

    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg, losses.avg


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
