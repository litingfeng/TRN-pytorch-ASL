import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool, Value
import argparse
import imageio
import os
from skimage import img_as_ubyte
import shutil
import time
import random


def get_existed_utt():
    flow_base_fold = base_fold + "utt_frame_of/"
    utt_list = os.listdir(flow_base_fold)
    valid_num = 0
    invalid_num = 0
    valid_list = []

    for cnt, utt in enumerate(utt_list):
        print("{} of {}".format(cnt + 1, len(utt_list)))
        flow_fold = flow_base_fold + "{}/".format(utt)
        frame_fold = base_fold + 'utt_frame/{}/'.format(utt)

        flow_list = os.listdir(flow_fold)
        flow_list = [frame for frame in flow_list if "flow_x" in frame]
        flow_num = len(flow_list)
        frame_num = len(os.listdir(frame_fold))

        if flow_num == (frame_num - 1):
            valid_num += 1
            valid_list.append(utt)
        else:
            #shutil.rmtree(flow_fold)
            invalid_num += 1
            print(utt)

    print("valid_num {}, invalid_num {}".format(valid_num, invalid_num))
    return valid_list


def norm_img(raw_flow,bound = 20):


    '''
    this function scale the input pixels to 0-255 with bi-bound
    '''
    flow = raw_flow

    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= -bound
    flow *= (255 / float(2*bound))
    return flow

def save_flows(flows, res_dir, f_id):

    #rescale to 0~255 with the bound setting
    flow_x=norm_img(flows[...,0])
    flow_y=norm_img(flows[...,1])

    #to numpy
    flow_x_img=Image.fromarray(flow_x) # only 1 channel
    flow_y_img=Image.fromarray(flow_y)

    #save img
    imageio.imwrite(res_dir + 'flow_x_{}.jpg'.format(f_id), np.uint8(flow_x_img))
    imageio.imwrite(res_dir + 'flow_y_{}.jpg'.format(f_id), np.uint8(flow_y_img))

def dense_flow(u_id):
    '''
    uid: utterance id
    '''
    # global counter
    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1

    print("{} of {}".format(counter.value, len(u_id_list)))

    src_dir = frame_fold + "{}/".format(u_id)
    res_dir = base_fold + "utt_frame_of/{}/".format(u_id)
    os.mkdir(res_dir)
    #u_id, save_dir,step,bound=augs

    frame_list = os.listdir(src_dir)
    frame_num = len(frame_list)

    for f_id in range(frame_num - 1):
        #load img
        start_time = time.time()
        prev_path = src_dir + "{}.png".format(f_id)
        this_path = src_dir + "{}.png".format(f_id + 1)

        #print(os.path.exists(prev_path))
        #print(os.path.exists(this_path))
        prev_gray = cv2.imread(prev_path, 0)
        this_gray = cv2.imread(this_path, 0)

        scale = 0.5  # percent of original size
        width = int(prev_gray.shape[1] * scale)
        height = int(prev_gray.shape[0] * scale)

        dim = (width, height)

        # resize image
        prev_gray = cv2.resize(prev_gray, dim, interpolation=cv2.INTER_AREA)
        this_gray = cv2.resize(this_gray, dim, interpolation=cv2.INTER_AREA)

        #extract of
        dtvl1 =cv2.optflow.DualTVL1OpticalFlow_create(nscales = 2, warps = 2, epsilon=0.02)
        flowDTVL1 = dtvl1.calc(prev_gray,this_gray,None)
        #print("computation time:", round(time.time() - start_time,3))
        start_time = time.time()
        save_flows(flowDTVL1, res_dir, f_id) #this is to save flows and img.
        #print("io time:", round(time.time() - start_time, 3))


if __name__ =='__main__':

    base_fold = "/dresden/gpu2/tl6012/data/ASL/high_resolution_data/"
    frame_fold = base_fold + "utt_frame/"
    u_id_list = os.listdir(frame_fold)

    existed_utt = [] #get_existed_utt()
    print(len(u_id_list))
    u_id_list = [utt for utt in u_id_list if utt not in existed_utt]
    print("there are {} utterances need to do".format(len(u_id_list)))
    #dense_flow(u_id_list[0])
    #'''
    counter = Value('i', 0)
    p = Pool(16, initargs=(counter,))
    p.map(dense_flow, u_id_list)
    #'''