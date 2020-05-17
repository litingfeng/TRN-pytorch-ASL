import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
import glob
from PIL import Image
import moviepy.editor as mpy
from shutil import copyfile

import torchvision
import torch.nn.parallel
import torch.optim
from models import TSN
import transforms
from torch.nn import functional as F


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='moments',
                    choices=['something', 'jester', 'moments', 'somethingv2', 'rachel', 'dai'])
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str)

args = parser.parse_args()

# Get dataset categories.
categories_file = '{}/category_label_20.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

checkpoint = torch.load(args.weights)
epoch      = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']
arch       = checkpoint['arch']
print('epoch {:d} best_prec1 {:.4f} arch {:s} '.format(epoch, best_prec1, arch)) # 80.7629
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupOverSample(net.input_size, net.scale_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(net.input_mean, net.input_std),
])

val_list = 'dai/val_videofolder_20.txt'
tmp = [x.strip().split(' ') for x in open(val_list)]
tmp = [item for item in tmp if int(item[1])>=3]
correct = 0
count  = 0
correct_class = [0]*num_class
count_class = [0]*num_class
save_dir = '/dresden/gpu2/tl6012/data/ASL/isolate_mistakes'
for c in categories:
    os.makedirs(os.path.join(save_dir,c), exist_ok=True)
# Obtain video frames
for frame_dir, _, label in tmp:
    frame_dir = os.path.join(args.frame_folder, frame_dir)

    # Here, make sure after sorting the frame paths have the correct temporal order
    frame_paths = sorted(glob.glob(frame_dir+'/*.png'))
    frames = load_frames(frame_paths, num_frames=3)

    # Make video prediction.
    data = transform(frames)
    input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda()

    try:
        with torch.no_grad():

            logits = net(input)

            h_x = torch.mean(F.softmax(logits, 1), dim=0).data
            probs, idx = h_x.sort(0, True)
            val ,inds = torch.max(h_x, 0)
            count += 1
    except:
        continue

    # Output the prediction.
    if inds != int(label):
        # print('WRONG!!!!!!')
        # print('input ', input.size())
        # print('logits ', logits.size())
        # print('hs=x ', h_x)
        # print('probs ', probs)
        # print('inds ', inds, '\n label ', label)
        # print('soft ', F.softmax(logits, 1).size())
        print('Loading frames in {}'.format(frame_dir))
        video_name = args.frame_folder if args.frame_folder is not None else args.video_file
        print('RESULT ON ' + video_name)
        for i in range(0, num_class):
            print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
        count_class[int(label)] += 1

        dst = os.path.join(save_dir, categories[inds.item()],
                           frame_dir.split('/')[-2]+'_'+frame_dir.split('/')[-1])
        print('dst ', dst)
        if not os.path.islink(dst):
            os.symlink(frame_dir, dst)
        #exit()

    else:
        correct += 1
        correct_class[int(label)] += 1
        count_class[int(label)] += 1
    #pred = torch.

print('acc ', correct / count, correct, count)
print('acc each class ', correct_class,'\n', count_class, '\n', np.array(correct_class)/np.array(count_class))

# Render output frames with prediction text.
if args.rendered_output is not None:
    prediction = categories[idx[0]]
    rendered_frames = render_frames(frames, prediction)
    clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
    clip.write_videofile(args.rendered_output)
