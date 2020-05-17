import os
import pdb
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np

dataset_name = 'dai'
data_root = '/dresden/gpu2/tl6012/data/ASL/isolated_signs/'

def select_class():
    labels = {}
    catfiles = os.listdir(data_root)
    categories = []
    for c in catfiles:
        instances = os.listdir(os.path.join(data_root, c))
        # frame number larger than 3 is a valid instance, must have > 20 instances
        num_ins = len([1 for i in instances
                   if len(os.listdir(os.path.join(data_root, c, i))) > 3])
        if num_ins > 20:
            categories.append(c)

    with open('./dai/category_label_20.txt','w') as f:
        f.write('\n'.join(categories))

def generate_lists():
    output_train = []
    output_val = []
    with open('./dai/category_label_20.txt','r') as f:
        categories = f.read().splitlines()

    for k, cat in enumerate(categories):
        data_cat = os.path.join(data_root, cat)
        assert(os.path.exists(data_cat))
        videos = [ i for i in os.listdir(data_cat) if len(os.listdir(os.path.join(data_cat, i)))>3]
        train_videos = np.random.choice(videos, int(len(videos)*0.8),replace=False)
        curFolders = [cat+'/'+v for v in train_videos]
        num_filess = [len(os.listdir(os.path.join(data_cat, i))) for i in train_videos]
        curIDXs    = [k] * len(train_videos)
        # [video_id, num_frames, class_idx]
        output_train += ['%s %d %d'%(curFolder, num_files, curIDX) for curFolder, num_files, curIDX in zip(curFolders,num_filess,curIDXs)]

        val_videos = [i for i in videos if i not in train_videos]
        curFolders = [cat + '/' + v for v in val_videos]
        num_filess = [len(os.listdir(os.path.join(data_cat, i))) for i in val_videos]
        curIDXs = [k] * len(val_videos)
        output_val += ['%s %d %d' % (curFolder, num_files, curIDX) for curFolder, num_files, curIDX in
                         zip(curFolders, num_filess, curIDXs)]

        print('train ', len(train_videos), ' test ', len(val_videos))

    with open('./dai/train_videofolder_20.txt','w') as f:
        f.write('\n'.join(output_train))
    with open('./dai/val_videofolder_20.txt','w') as f:
        f.write('\n'.join(output_val))

if __name__ == '__main__':

    select_class()
    generate_lists()