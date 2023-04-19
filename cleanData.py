import cv2
import os
import numpy as np
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path', type=str, help='folder path to clean and filter')
    parser.add_argument('--color', type=bool, default=False, help='filter almost all black, white, grey patches')
    parser.add_argument('--marks', type=bool, default=False, help='filter weird human-annotated marks on patches that may not be relevant')
    return parser.parse_args()

def filter(args): #input: folder path; output: # of images removed
    print(f"cleaning {os.path.abspath(args.path)}")
    path = args.path
    file_names = os.listdir(path)
    bad_set = set()
    for f in file_names:
        imgPath = os.path.join(path,f)
        if f in bad_set or os.path.isdir(imgPath):
            continue
        img = cv2.imread(imgPath, 1)
        
        if args.color:
            temp = np.copy(img)
            rgb_mean = np.mean(temp)
            if (rgb_mean > 230) or (rgb_mean < 10) or (rgb_mean == None):
                bad_set.add(f)
        
        if args.marks:
            r = np.std(img[:,:,0])
            g = np.std(img[:,:,1])
            b = np.std(img[:,:,2])
            rPerct = r / (r+g+b)
            gPerct = g / (r+g+b)
            bPerct = b / (r+g+b)
            stdBw = np.std(np.array([rPerct,gPerct,bPerct]))

            if bPerct > 0.46:
                bad_set.add(f)
            if np.round(stdBw,2) == 0.0:
                bad_set.add(f)

    if not os.path.exists(os.path.join(path,'bad_folder')):
        os.makedirs(os.path.join(path,'bad_folder'))
    bad_list = list(bad_set)
    bad_list.sort()
    for im in bad_list:
        old_fp = os.path.join(path, im)
        new_fp = os.path.join(path,'bad_folder',im)
        shutil.copyfile(old_fp, new_fp)
        os.remove(old_fp)
    print("==== filter images ====")
    print("  - removing", len(bad_list), "images (moved to bad_folder. Check and delete as needed)")
    return len(bad_list)

if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.path):
        filter(args)
    else:
        print("Below path doesn't exist: (put this file in same folder)")
        print(os.path.abspath(args.path))