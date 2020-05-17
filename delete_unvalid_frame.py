import os

def delete_unvalid_frame(utt):
    frame_fold = base_fold + "utt_frame/{}/".format(utt)
    frame_list = os.listdir(frame_fold)
    unvalid_frame = []
    for frame in frame_list:
        img_pth = frame_fold + frame
        size = os.path.getsize(img_pth) / 1000
        #print(frame,size)
        #print(size)
        if size <= 20:
            print(frame)
            if frame == "0.png":
                #print('zero ', utt)
                raise ValueError(utt)
            else:
                os.remove(img_pth)


if __name__ == "__main__":
    base_fold = "/dresden/gpu2/tl6012/data/ASL/high_resolution_data/"

    utt_list = os.listdir(os.path.join(base_fold, 'utt_frame'))
    for utt in utt_list:
        delete_unvalid_frame(utt)