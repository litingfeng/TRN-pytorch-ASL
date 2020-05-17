import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import numpy.random as random
import torch
from transforms import *

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def handshapes(self): # r_start, l_start, r_end, l_end --> l_s, l_e, r_s, r_e
        #print(self._data)
        handshapes = [int(i) for i in self._data[3:]]
        return [handshapes[1], handshapes[3], handshapes[0], handshapes[2]]


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 test_mode=False, siamese=False, hand=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.siamese = siamese
        self.hand = hand

        if self.hand:
            self.handbox_dir = '/dresden/gpu2/tl6012/data/ASL/dai_lexical_handbox/'
            self.handbox_tmpl = '{:d}.npy'

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def _load_image_hand(self, directory, base_idx, numframes):

        # boxes are in left right order, or just one box
        boxes = [np.load(os.path.join(self.handbox_dir, directory,
                              self.handbox_tmpl.format(i))) for i in [base_idx, base_idx+numframes-1]]
        #print('boxes0 ', boxes)
        # open the start & end frame
        frames = [Image.open(os.path.join(self.root_path, directory,
                                          self.image_tmpl.format(i))) for i in [base_idx, base_idx + numframes - 1]]

        boxes = [ np.vstack((b,b)) if b.shape[0]==1 else b for b in boxes ]

        boxes = [ np.array((int(max(0, h[0])), int(max(h[1], 0)), int(min(h[2], frames[i].width-1)), int(min(h[3],frames[i].height-1))))
                           for i,b in enumerate(boxes) for h in b]

        boxes = [b[i-1]  if not np.any(b) else b for i,b in enumerate(boxes)]


        # start_left, start_right, end_left, end_right
        new_frames = [frames[0].crop(box=b) if i < 2 else frames[1].crop(b)
                      for i, b in enumerate(boxes)]

        # change order to start_left, end_left, start_right, end_right
        new_frames = [new_frames[0], new_frames[2], new_frames[1], new_frames[3]]

        return new_frames


    def _load_image(self, directory, idx, base_idx):

        img_list = []
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            idx = idx + base_idx  # only for rachel
            try:
                img_list += [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
                #print('idx ', idx,'\t',  os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                img_list += [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(base_idx))).convert('RGB')]
        if 'Flow' in self.modality:
            try:
                idx_skip = idx + base_idx #1 + idx*5 + base_idx
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
                #print('idx flow ', idx,'\t',  os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                print('idx ',idx)
                #exit()
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(base_idx))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            img_list += [x_img, y_img]

        return img_list

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx, right_start_hs, left_start_hs, right_end_hs, left_end_hs]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if 'Flow' in self.modality:
            tmp = [[x[0], str(int(x[1])-1), x[2]] for x in tmp]
        tmp = [item for item in tmp if int(item[1])>=3]

        self.video_list = [VideoRecord(item) for item in tmp]
        self.labels     = [int(item[2]) for item in tmp]
        print('video number:%d'%(len(self.video_list)))


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        #print('mod ', self.modality, record.path)
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        if self.siamese:
            # we need to make sure approx 50% of images are in the same class
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                while True:
                    # keep looping till the same class video is found
                    record2 = self.video_list[random.choice(range(len(self.video_list)))]
                    if record.label == record2.label:
                        break
            else:
                while True:
                    # keep looping till a different class image is found
                    record2 = self.video_list[random.choice(range(len(self.video_list)))]
                    if record.label != record2.label:
                        break

            if not self.test_mode:
                segment_indices2 = self._sample_indices(record2) if self.random_shift else self._get_val_indices(record2)
            else:
                segment_indices2 = self._get_test_indices(record2)

            process_data, label = self.get(record, segment_indices)
            process_data2, label2 = self.get(record2, segment_indices2)

            return process_data, process_data2, torch.from_numpy(np.array([int(label != label2)], dtype=np.float32))

        else:
            return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        #print('path ', record.path)
        if record.num_frames < 3:
            print('not enough of')
            exit()

        p_tmp = int(record.path.split('/')[1].split('_')[1])
        for seg_ind in indices: # indices starts from 1
            p = int(seg_ind)-1

            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p, p_tmp)
                images.extend(seg_imgs)
                if p+1 < record.num_frames:
                    p += 1

        images_hand = self._load_image_hand(record.path, p_tmp, record.num_frames)

        process_data = self.transform(images)
        process_data_hand = self.transform(images_hand)

        return process_data, record.label, process_data_hand, np.array(record.handshapes)


    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    trainset = TSNDataSet('/dresden/gpu2/tl6012/data/ASL/isolated_signs',
                          'dai/train_videofolder_20_hand.txt',
                          num_segments=2,
                          new_length=1,
                          modality='RGB',
                          image_tmpl='{:d}.png',
                          transform=torchvision.transforms.Compose([
                              GroupScale(int(224 * 256 // 224)),
                              GroupCenterCrop(224),
                              Stack(roll=True),
                              ToTorchFormatTensor(div=(False)),
                              GroupNormalize([104, 117, 128], [1]),
                          ]),
                          hand=True)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2, shuffle=True,
        num_workers=0, pin_memory=True)

    for i, data in enumerate(train_loader):
        input1, target = data
        print(input1.size())
        print('target ', target)
        input = input1.view(-1, 6, input1.size(2), input1.size(3))
        target = target.view(-1, 2)
        print(input.size())
        print(target)
        exit()
        continue
