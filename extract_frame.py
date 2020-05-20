import cv2
import os
from multiprocessing import Pool, Value

def get_name_list_from_data():
    base_fold = "/dresden/gpu2/tl6012/data/ASL/2015-movies/"
    fold_list = os.listdir(base_fold)
    fold_list = [ele for ele in fold_list if "Zhiyi" not in ele]
    all_name = []
    for fold in fold_list:
        src_pth = base_fold + fold
        name_list = os.listdir(src_pth)
        name_list = [src_pth + "/" + name for name in name_list if "cam1" in name]
        all_name += name_list

    return all_name

def extract_frame(video_pth):

    global counter
    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1

    print("{} of {}".format(counter.value, len(valid_name_list)))

    # pens the Video file
    base_res_fold = "/dresden/gpu2/tl6012/data/ASL/high_resolution_data/all_frame/"
    video_name = video_pth.split("/")[-1].split(".mp4")[0]
    res_fold = base_res_fold + video_name + "/"
    os.mkdir(res_fold)
    cap = cv2.VideoCapture(video_pth)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(res_fold + str(i) + '.png', frame)
        i += 1


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    valid_name_list = get_name_list_from_data()

    counter = Value('i', 0)
    p = Pool(32, initargs=(counter,))

    # frame2video(fold_name_list[0])

    p.map(extract_frame, valid_name_list)