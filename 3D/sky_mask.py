import cv2
from scipy.signal import medfilt
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

def cal_skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20:
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask

def get_sky_region_gradient(img):
    h, w, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (9, 3))
    cv2.medianBlur(img_gray, 5)
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)
    gradient_mask = (lap < 3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)
    # plt.imshow(mask)
    # plt.show()
    mask = cal_skyline(mask)
    #after_img = cv2.bitwise_and(img, img, mask=mask)
    return mask

import cv2
from matplotlib import pyplot as plt 
import os
from os import listdir
from os.path import isfile, join

def draw(prefix):
    for file in sorted([f for f in listdir(prefix) if f.endswith('png') and 'mask' not in f]):
        #img = cv2.imread(""/mnt/nas/kaichen/compete/png/scene1/0193.png"")[:,:,::-1]
        print(prefix, file)
        index = file.split('/')[-1].split('.')[0]
        img = cv2.imread(os.path.join(prefix, file))[:,:,::-1]
        img_sky = get_sky_region_gradient(img)
        mask = (1 - img_sky)
        mask = np.concatenate((mask[:400], np.ones((680,1920))), axis=0)
        cv2.imwrite(os.path.join(prefix, "mask_{}.png".format(index)), mask*255.)

prefix = "/mnt/nas/kaichen/compete/png/scene1/"
draw(prefix)
prefix = "/mnt/nas/kaichen/compete/png/scene2/"
draw(prefix)
