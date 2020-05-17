import numpy as np
import json
import os
from shutil import copyfile, copytree, Error, move, rmtree
from PIL import Image
from multiprocessing import Pool

base_fold    = '/dresden/gpu2/tl6012/data/ASL/'
ann_fold     = base_fold + 'manual_signs_label_969/'
rgb_fold     = os.path.join(base_fold, 'high_resolution_data/utt_frame')

def load_txt(file_path):
  file = open(file_path)
  result = []

  for line in file:
    result.append(line.split()[0])
  return result


def load_json(file):
  with open(file) as json_file:
    data = json.load(json_file)
    return data

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_intervals(Yi):
  idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
  intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
  return intervals


def parse_ann(uid):
  ann_path = ann_fold + "{}.json".format(uid)
  ann_data = load_json(ann_path)['label']

  return ann_data


def copy_one(uid):
    # parse anno
    ann = parse_ann(uid)

    # get segmentation-level annation
    ann = list(ann.values())
    intervals = segment_intervals(ann)
    lab = segment_labels(ann)
    ind = np.nonzero(lab)[0]
    lab = lab[ind]
    ann = [intervals[i] for i in ind] # a list of (start_frame, end_frame) tuple
    segment = {}

    for seg, l in zip(ann, lab):
      if not uid in segment.keys():
        segment[uid] = [[seg[0], seg[1]-1, l]]
      else:
        segment[uid].append([seg[0], seg[1]-1, l])

      if l in idx2cls.keys():
          save_dir = os.path.join(save_root, idx2cls[l],uid+'_'+str(seg[0]))
          print('save to ', save_dir)
          os.makedirs(save_dir, exist_ok=True)



          for frame in range(seg[0], seg[1]):
            # rgb
            src = os.path.join(rgb_fold, uid, str(frame)+'.png')
            dst = os.path.join(save_dir, str(frame)+'.png')
            copyfile(src, dst)


        # of
        # if os.path.exists(os.path.join(of_fold, uid, 'flow_x_'+str(frame)+'.jpg')):
        #     flow_x  = np.array(Image.open(os.path.join(of_fold, uid, 'flow_x_'+str(frame)+'.jpg')))
        # else:
        #     continue
        #
        # if os.path.exists(os.path.join(of_fold, uid, 'flow_y_'+str(frame)+'.jpg')):
        #     flow_y  = np.array(Image.open(os.path.join(of_fold, uid, 'flow_y_'+str(frame)+'.jpg')))
        #
        # blank_img = np.zeros(np.array(flow_x).shape, 'uint16')
        # new_flow  = np.stack((flow_x, flow_y, blank_img), axis=2)
        # new_flow_img = Image.fromarray(new_flow.astype(np.uint8))
        # new_flow_img.save(os.path.join(save_dir, 'frame'+str(frame)+'.jpg'))


if __name__ == '__main__':
    idx2cls = {}
    with open('vocabulary.txt', 'r') as filehandle:
        vocabulary = filehandle.read().splitlines()
        for i, v in enumerate(vocabulary):
            vocab = v.split("'")
            if '/' in vocab[1]:
                vocab[1] = vocab[1].replace('/', 'or') # replace '/' in sign label
            idx2cls[i+1] = vocab[1]
    save_root = '/dresden/gpu2/tl6012/data/ASL/isolated_signs/'
    os.makedirs(save_root, exist_ok=True)
    data_split = os.listdir(rgb_fold)

    max_lab = 0
    #copy_one(data_split[0])

    process_num = 8

    p = Pool(process_num)
    p.map(copy_one, data_split)
    print('finished')