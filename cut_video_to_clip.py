import xml.etree.ElementTree as ET
import json
import os
import shutil
from multiprocessing import Pool, Value

def get_start_end_frame(element):
    s_frame = int(element.get("START_FRAME")) / 1001
    e_frame = int(element.get("END_FRAME")) / 1001
    s_frame = round(s_frame)
    e_frame = round(e_frame)

    return (s_frame, e_frame)

def get_name_list_from_data():
    base_fold = "/dresden/gpu2/tl6012/data/ASL/2015-movies/"
    fold_list = os.listdir(base_fold)
    fold_list = [ele for ele in fold_list if "Zhiyi" not in ele]
    all_name = []
    for fold in fold_list:
        src_pth = base_fold + fold
        name_list = os.listdir(src_pth)
        name_list = [name for name in name_list if "cam1" in name]
        all_name += name_list

    return all_name

def get_utt_img(param):
    utt, valid_name = param
    global counter
    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1

    print("{} of {}".format(counter.value, len(valid_name_list)))

    #set src fold
    base_src_fold = "/dresden/gpu2/tl6012/data/ASL/high_resolution_data/all_frame/{}/"
    valid_name = valid_name.split(".mp4")[0]
    src_fold = base_src_fold.format(valid_name)
    #set target fold
    utt_id = utt.get("ID")
    u_s_frame, u_e_frame = get_start_end_frame(utt)
    res_fold = base_res_fold + "{}/".format(utt_id)
    os.makedirs(res_fold,exist_ok=True)


    #copy image from src to target
    tgt_idx = 0
    for i in range(u_s_frame, u_e_frame + 1):
        src_pth = src_fold + "{}.png".format(i)
        if not os.path.exists(src_pth):
            continue
        tgt_pth = res_fold + "{}.png".format(tgt_idx)
        shutil.copyfile(src_pth, tgt_pth)
        tgt_idx += 1





if __name__ == "__main__":
    valid_name_list = get_name_list_from_data()
    counter = Value('i', 0)

    base_res_fold = "/dresden/gpu2/tl6012/data/ASL/high_resolution_data/utt_frame/"
    src_path = "/ajax/users/tl601/projects/asl/969_71106.xml" #the pth for xml file
    tree = ET.parse(src_path)
    root = tree.getroot()

    c_list = root.findall("./COLLECTIONS/")

    src_path = "/ajax/users/tl601/projects/asl/969_71106.xml"  # the pth for xml file
    tree = ET.parse(src_path)
    root = tree.getroot()

    c_list = root.findall("./COLLECTIONS/")
    all_utt_num = 0
    for c_ele in c_list:
        meta_files = c_ele.findall("./MEDIA-FILES/")
        #check weather we
        is_valid = False #flag indicating whether we have ne videor
        valid_name = None #video name
        for meta_file in meta_files:
            for neighbor in meta_file.iter('MEDIA-FILE'):
                video_name = neighbor.get("FILE-NAME")
                if video_name in valid_name_list:
                    valid_name = video_name
                    is_valid = True

        if is_valid:
            ann_info = c_ele.findall("./TEMPORAL-PARTITIONS/TEMPORAL-PARTITION/SEGMENT-TIERS/SEGMENT-TIER/")
            #print(len(ann_info))
            assert len(ann_info) == 3
            signer = ann_info[0].text
            utt_list = ann_info[2].findall("UTTERANCE")

            p = Pool(32, initargs=(counter,))

            input_list = [(utt, valid_name) for utt in utt_list]

            p.map(get_utt_img, input_list)