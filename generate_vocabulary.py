import os
import pdb
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import math
import itertools

# def cal_frame(frame):
#     return int(math.ceil(eval(frame)/1001))

def cal_frame(frame):
    return round(int(eval(frame))/1001)


def get_node_value(element, name):
    if len(element.getElementsByTagName(name)[0].childNodes)>0:
        return element.getElementsByTagName(name)[0].childNodes[0].nodeValue
    else:
        return None

rbgfiles = '/dresden/gpu2/tl6012/data/ASL/high_resolution_data/utt_frame/'
DOMTree = xml.dom.minidom.parse('969_71106.xml')
collection= DOMTree.documentElement
utterances=collection.getElementsByTagName('UTTERANCE')

#ab = {'61830117':25974, '4236780':15276, '17583211':6957, '36287804':4999}

# nonconsistent = ['633633', '61830117', '4236780', '17583211', '26852527', '519141040',
#                  '36287804', '978522', '33733139', '33733928', '33734224', '33734520', '33734816']
sign_dict = {}
num = {}
for j, utterance in enumerate(utterances):
    ut = {}
    a = utterance.getElementsByTagName('MANUALS')
    uid = utterance.getAttribute('ID')

    # start and end of utterance
    stf = cal_frame(utterance.getAttribute('START_FRAME'))
    edf = cal_frame(utterance.getAttribute('END_FRAME'))

    ut['utterance_id'] = uid
    ut['start_frame'] = stf
    ut['end_frame'] = edf

    ann_frames = edf - stf + 1
    if os.path.exists(rbgfiles + uid):
        files = os.listdir(rbgfiles + uid)
        #num[abs(ann_frames - len(files))]
        if ann_frames != len(files):
            print(uid, ' ', ann_frames, ' ', len(files))
            edf = stf + len(files) - 1
            ut['end_frame'] = edf
            if abs(ann_frames - len(files)) > 1:
                print('\n', uid, ' ', ann_frames, ' ', len(files))
        assert(edf - stf + 1 == len(files))

    signs=a[0].getElementsByTagName('SIGN')
    labels = {}
    for sign in signs:
        ID = sign.getAttribute('ID')
        LABEL=get_node_value(sign,'LABEL')
        SIGN_TYPE = get_node_value(sign, 'SIGN_TYPE')
        if SIGN_TYPE != "'Lexical Signs'":
            continue
        TWO_HANDED = get_node_value(sign, 'TWO_HANDED')
        D_START_HS = get_node_value(sign, 'D_START_HS')
        ND_START_HS = get_node_value(sign, 'ND_START_HS')
        D_END_HS = get_node_value(sign, 'D_END_HS')
        ND_END_HS = get_node_value(sign, 'ND_END_HS')
        TWOHANDED_HANDSHAPES = get_node_value(sign, 'TWOHANDED_HANDSHAPES')
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
        sign_dict[uid+"_"+str(start-stf)] = [LABEL]#, D_START_HS, ND_START_HS, D_END_HS, ND_END_HS]



vocabulary = set([v[0] for k,v in sign_dict.items()])

with open('dai/vocabulary.txt','w') as f:
    f.write('\n'.join(vocabulary))

handshape = [v[1:] for k, v in sign_dict.items()]

handshape = list(set(list(itertools.chain(*handshape))))

with open('dai/handshape.txt', 'w') as f:
    f.write('\n'.join(handshape))









