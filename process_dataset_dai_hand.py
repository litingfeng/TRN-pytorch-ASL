
import os
import pdb
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import math
import itertools

def cal_frame(frame):
    return round(int(eval(frame))/1001)

def get_node_value(element, name):
    if len(element.getElementsByTagName(name)[0].childNodes)>0:
        return element.getElementsByTagName(name)[0].childNodes[0].nodeValue
    else:
        return None

dataset_name = 'dai'
rbgfiles = '/dresden/gpu2/tl6012/data/ASL/isolated_signs/'

with open('./dai/handshape.txt', 'r') as filehandle:
    handshape = filehandle.read().splitlines()
with open('./dai/vocabulary.txt', 'r') as filehandle:
    vocabulary = filehandle.read().splitlines()
with open('./dai/train_videofolder_20.txt','r') as f:
    train_videofolder = f.read().splitlines()
with open('./dai/val_videofolder_20.txt','r') as f:
    val_videofolder = f.read().splitlines()

handshape2idx = {}
for i, h in enumerate(handshape):
    handshape2idx[h] =  i

vocabulary2idx = {}
for i, h in enumerate(vocabulary):
    vocabulary2idx[h] =  i


#print('rache ', rachel_uid)
# parse xml file to get handshape label
DOMTree = xml.dom.minidom.parse('969_71106.xml') # 1908 uid, 1810 unique uid
collection= DOMTree.documentElement
utterances=collection.getElementsByTagName('UTTERANCE')

sign_dict = {}
for j, utterance in enumerate(utterances):
    ut = {}
    a = utterance.getElementsByTagName('MANUALS')
    uid = utterance.getAttribute('ID')
    # if uid != '15720711':
    #     continue

    # start and end of utterance
    stf = cal_frame(utterance.getAttribute('START_FRAME'))
    edf = cal_frame(utterance.getAttribute('END_FRAME'))

    ut['utterance_id'] = uid
    ut['start_frame'] = stf
    ut['end_frame'] = edf

    ann_frames = edf - stf + 1
    if os.path.exists(rbgfiles + uid):
        files = os.listdir(rbgfiles + uid)
        if ann_frames != len(files):
            print(uid, ' ', ann_frames, ' ', len(files))
            edf = stf + len(files) - 1
            ut['end_frame'] = edf
            if abs(ann_frames - len(files)) > 1:
                print('\n', uid, ' ', ann_frames, ' ', len(files))
        assert (edf - stf + 1 == len(files))

    signs=a[0].getElementsByTagName('SIGN')
    labels = {}
    for sign in signs:
        ID = sign.getAttribute('ID')
        LABEL=get_node_value(sign,'LABEL')
        SIGN_TYPE = get_node_value(sign, 'SIGN_TYPE')
        if SIGN_TYPE != "'Lexical Signs'":
            continue
        #TWO_HANDED = get_node_value(sign, 'TWO_HANDED')
        D_START_HS = get_node_value(sign, 'D_START_HS')
        ND_START_HS = get_node_value(sign, 'ND_START_HS')
        D_END_HS = get_node_value(sign, 'D_END_HS')
        ND_END_HS = get_node_value(sign, 'ND_END_HS')
        # TWOHANDED_HANDSHAPES = get_node_value(sign, 'TWOHANDED_HANDSHAPES')
        DOMINANT_HAND = sign.getElementsByTagName('DOMINANT_HAND')
        INITIAL_HOLD = sign.getElementsByTagName('INITIAL_HOLD')
        FINAL_HOLD = sign.getElementsByTagName('FINAL_HOLD')
        start = DOMINANT_HAND[0].getAttribute('START_FRAME')
        end = DOMINANT_HAND[0].getAttribute('END_FRAME')
        if INITIAL_HOLD:
            start = INITIAL_HOLD[0].getAttribute('START_FRAME')
        if FINAL_HOLD:
            end = FINAL_HOLD[0].getAttribute('END_FRAME')
        start = cal_frame(start)
        end = cal_frame(end)

        LABEL = LABEL.replace('/', 'or').split("'")[1]
        # if LABEL == '(1h)GOODorTHANK-YOU':
        #print('key ', LABEL+'/'+uid+"_"+str(start-stf), str(end-stf))
        sign_dict[LABEL+'/'+uid+"_"+str(start-stf)] = [handshape2idx[D_START_HS], handshape2idx[ND_START_HS],
                                                   handshape2idx[D_END_HS], handshape2idx[ND_END_HS]]

for i, item in enumerate(train_videofolder):
    k = item.split(' ')[0]
    # deal with one frame has duplicately labeled when parsing
    if k == 'IX-1p/3378363_99':
        k = 'IX-1p/3378363_98'
    elif k == '(1h)GOODorTHANK-YOU/8395276_150':
        k = '(1h)GOODorTHANK-YOU/8395276_145'
    handshapes = sign_dict[k]
    train_videofolder[i] = '{:s} {:d} {:d} {:d} {:d}'.format(item, handshapes[0], handshapes[1], handshapes[2], handshapes[3])

for i, item in enumerate(val_videofolder):
    k = item.split(' ')[0]
    if k == 'IX-3p:i/7986521_105':
        k = 'IX-3p:i/7986521_104'
    elif k == '(1h)TEND/15720711_84':
        k = '(1h)TEND/15720711_82'
    handshapes = sign_dict[k]
    val_videofolder[i] = '{:s} {:d} {:d} {:d} {:d}'.format(item, handshapes[0], handshapes[1], handshapes[2], handshapes[3])

# each row: [video_id, num_frames, class_idx, right_start_hs, left_start_hs, right_end_hs, left_end_hs]
with open('./dai/train_videofolder_20_hand.txt','w') as f:
    f.write('\n'.join(train_videofolder))
with open('./dai/val_videofolder_20_hand.txt','w') as f:
    f.write('\n'.join(val_videofolder))



