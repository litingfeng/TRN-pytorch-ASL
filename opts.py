import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, choices=['something','jester','moments', 'somethingv2', 'rachel', 'dai'])
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str,default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--data_length', default=3, type=int,
                    help='length of stacked optical flow images')

parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll', 'CB'])
parser.add_argument('--loss_subtype', type=str, default="focal",
                    choices=['focal', 'softmax', 'sigmoid'])
parser.add_argument('--gamma', default=1.0, type=float,
                    help='for class balanced loss parameter')
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")

parser.add_argument('--hand', default=False, action="store_true", help='use hand bounding boxes')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--lamdb_hand', default=1.0, type=float,
                    help='weight of handshape loss')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_of', default='', type=str,
                    help='path to latest optical flow checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='/dresden/gpu2/tl6012/TRN-test/log')
parser.add_argument('--root_model', type=str, default='/dresden/gpu2/tl6012/TRN-test/model')
parser.add_argument('--root_output',type=str, default='/dresden/gpu2/tl6012/TRN-test/output')


# ======================= Siamese Configs =============================
parser.add_argument('--siamese', default=False, action='store_true',
                        help='turn on siamese network for classification')
parser.add_argument('--margin', default=1.5, type=float,
                    help='margin for contrastive loss')



