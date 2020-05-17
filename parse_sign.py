from xml.dom.minidom import parse
import xml.dom.minidom
import json
import numpy as np
import scipy.io as scio
import os


def get_node_value(element, name):
    if len(element.getElementsByTagName(name)[0].childNodes)>0:
        return element.getElementsByTagName(name)[0].childNodes[0].nodeValue
    else:
        return None

def cal_frame(frame):
    return round(int(eval(frame))/1001)


def search_classes():
    DOMTree = xml.dom.minidom.parse('969_71106.xml')
    collection = DOMTree.documentElement
    utterances = collection.getElementsByTagName('UTTERANCE')
    labels = {}
    sin_num = 0
    for utterance in utterances:

        a = utterance.getElementsByTagName('MANUALS')

        signs = a[0].getElementsByTagName('SIGN')
        for sign in signs:
            SIGN_TYPE = get_node_value(sign, 'SIGN_TYPE')
            if SIGN_TYPE == "'Lexical Signs'":
                sin_num += 1
                LABEL = get_node_value(sign, 'LABEL')
                if LABEL not in labels:
                    labels[LABEL] = 1
                else:
                    labels[LABEL] += 1

    # w = csv.writer(open("lexical_label_statistics_969.csv", "w"))
    count = 0
    for key, val in labels.items():
        if val > 25:
            count += 1
            # print(key, val)
        # w.writerow([key, val])
    return labels


def parse_signs_label():
    '''
    Generate /dresden/gpu2//tl6012/data/ASL/manual_signs_label:
    only lexical signs are
    :return:
    '''
    DOMTree = xml.dom.minidom.parse('969_71106.xml')
    collection= DOMTree.documentElement
    utterances=collection.getElementsByTagName('UTTERANCE')
    base_dir = '/dresden/gpu2//tl6012/data/ASL/manual_signs_label_969/'
    os.makedirs(base_dir, exist_ok=True)

    with open('vocabulary.txt', 'r') as filehandle:
        vocabulary = filehandle.read().splitlines()

    label2idx = {}
    for i, v in enumerate(vocabulary):
        label2idx[v] = i+1

    for j, utterance in enumerate(utterances):
        ut = {}
        a = utterance.getElementsByTagName('MANUALS')
        uid = utterance.getAttribute('ID')
        # if uid != '8395276':
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

        label_per_frame = {}
        for i in range(stf, edf + 1):
            key = str(i)
            label_per_frame[key] = 0

        signs=a[0].getElementsByTagName('SIGN')
        labels = {}
        for sign in signs:
            ID = sign.getAttribute('ID')
            LABEL=get_node_value(sign,'LABEL')
            SIGN_TYPE = get_node_value(sign, 'SIGN_TYPE')
            TWO_HANDED = get_node_value(sign, 'TWO_HANDED')
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

            if SIGN_TYPE == "'Lexical Signs'":
                #LABEL = LABEL.replace('/', 'or').split("'")[1]
                if LABEL in label2idx.keys():
                    for i in range(start, end + 1):
                        label_per_frame[str(i)] = label2idx[LABEL]

        ut['label'] = label_per_frame
        save_name = base_dir + uid + '.json'
        out_file = open(save_name, 'w')
        json.dump(ut, out_file)
        print(j, ' ', save_name)


if __name__ == '__main__':
    rbgfiles = '/dresden/gpu2/tl6012/data/ASL/high_resolution_data/utt_frame/'
    parse_signs_label()




