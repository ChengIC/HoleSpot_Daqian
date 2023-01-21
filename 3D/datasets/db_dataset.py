# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset
from collections import Counter
from PIL import Image

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[1., 0, 0, 0],
                           [0, 1., 0, 0],
                           [0, 0, 1., 0],
                           [0, 0, 0, 1.]], dtype=np.float32)
        self.full_res_shape = (1920, 1080) # (1920, 1080) -> (1920, 400:1080)
    def index_to_folder_and_frame_idx(self, index):
        frame_index = int(self.filenames[index].split('.')[0])
        return frame_index
    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        #color = Image.fromarray(np.array(color)[400:])
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

class DBRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""
    def __init__(self, *args, **kwargs):
        super(DBRAWDataset, self).__init__(*args, **kwargs)
    def get_image_path(self, folder, frame_index):
        f_str = "{:04d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            folder, f_str)
        return image_path

def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass
    return data

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1