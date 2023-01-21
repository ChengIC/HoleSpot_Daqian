import cv2
import numpy as np
import glob
from os import listdir

def video_make(prefix, name):
    print(prefix, '============', name)
    if 'mask' in name:
        LIST = sorted([f for f in listdir(prefix) if f.endswith('png') and 'mask' in f])
    else:
        LIST = sorted([f for f in listdir(prefix) if f.endswith('png') and 'mask' not in f])
    print(LIST)
    img_array = []
    for filename in LIST:
        img = cv2.imread(prefix+filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter('video/{}.mp4'.format(name), cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    del img_array

if __name__ == '__main__':
    """
    prefix = "/mnt/nas/kaichen/compete/png/scene1/"
    name = 'scene1'
    video_make(prefix, name)
    prefix = "/mnt/nas/kaichen/compete/png/scene2/"
    name = 'scene2'
    video_make(prefix, name)
    prefix = "/mnt/nas/kaichen/compete/png/scene1/"
    name = 'mask1'
    video_make(prefix, name)
    prefix = "/mnt/nas/kaichen/compete/png/scene2/"
    name = 'mask2'
    video_make(prefix, name)
    """
    prefix = "/mnt/nas/kaichen/eng/COMPETE/DIFFNET/mono_model/log22/depth1/"
    name = 'depth1'
    video_make(prefix, name)
    prefix = "/mnt/nas/kaichen/eng/COMPETE/DIFFNET/mono_model/log22/depth2/"
    name = 'depth2'
    video_make(prefix, name)
    """
    prefix = "/mnt/nas/kaichen/eng/COMPETE/DIFFNET/mono_model/log22/taj1/"
    name = 'trajectory1'
    video_make(prefix, name)
    prefix = "/mnt/nas/kaichen/eng/COMPETE/DIFFNET/mono_model/log22/taj2/"
    name = 'trajectory2'
    video_make(prefix, name)
    """
    #tar -czvf trajectory1.tar.gz taj1
    #tar -czvf trajectory2.tar.gz taj2
    #tar -czvf depth1.tar.gz depth1
    #tar -czvf depth2.tar.gz depth2
