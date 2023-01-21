from __future__ import absolute_import, division, print_function
from datetime import datetime
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import torchvision
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
import cv2
import matplotlib.pyplot as plt
MIN_DEPTH = 0
MAX_DEPTH = 100
from tqdm import tqdm
def disp2rgb(disp):
    H = disp.shape[0]
    W = disp.shape[1]
    I = disp.flatten()
    map = np.array([[0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174], [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]])
    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]
    ind = np.minimum(np.sum( np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:,None], I.shape[0], axis=1), axis=0), 6)
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:,None])
    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(np.maximum(np.multiply(map[ind,0:3], np.repeat(1-I[:,None], 3, axis=1))\
         + np.multiply(map[ind+1,0:3], np.repeat(I[:,None], 3, axis=1)),0),1)
    I = np.reshape(I, [H, W, 3]).astype(np.float32)
    return I

class Evaler:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join('./', self.opt.model_name, self.opt.log_name)
        print('---------------------', self.log_path)
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        self.use_pose_net = True
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        #self.models["encoder"] = networks.test_hr_encoder.hrnet18(True)
        self.models["encoder"] = networks.hrnet_encoder.hrnet18(True)
        self.models["encoder"].num_ch_enc = [64, 18, 36, 72, 144]
        para_sum = sum(p.numel() for p in self.models['encoder'].parameters())
        print('params in encoder',para_sum)
        #self.models["depth"] = networks.HRDepthDecoder(
        #    self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"] = networks.DepthDecoder_MSF(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        para_sum = sum(p.numel() for p in self.models['depth'].parameters())
        
        print('params in depth decdoer',para_sum)
        if self.use_pose_net: #use_pose_net = True
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose_encoder"].cuda()
            self.models["pose"].cuda()
            self.parameters_to_train += list(self.models["pose"].parameters())
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        if self.opt.load_weights_folder is not None:
            print('-----LOADING-----')
            self.load_model()
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # splits --- eigen_zhou
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames_k = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames_k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        # ----------- VALIDATION ----------
        val_dataset1 = datasets.DBRAWDataset(
            self.opt.data_path, 'scene1', self.opt.height, self.opt.width, [0, 1], 4, is_train=False, img_ext='.png')
        self.val_loader1 = DataLoader(
            val_dataset1, 1, False, num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=False)
        val_dataset2 = datasets.DBRAWDataset(
            self.opt.data_path, 'scene2', self.opt.height, self.opt.width, [0, 1], 4, is_train=False, img_ext='.png')
        self.val_loader2 = DataLoader(
            val_dataset2, 1, False, num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=False)
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale) # defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        print("Using split:\n  ", self.opt.split)

    def set_train(self):
        """Convert all models to training mode"""
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def eval(self):
        self.init_time = time.time()
        self.step = 0
        self.start_time = time.time()
        self.epoch = 0
        self.test_epoch(self.val_loader1, 'mono_model/{}/taj1'.format(self.opt.log_name))
        self.test_epoch(self.val_loader2, 'mono_model/{}/taj2'.format(self.opt.log_name))
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def test_epoch(self, data_load, path):
        print("============> Validation{} <============".format(self.epoch))
        if not os.path.exists(path):
            os.makedirs(path) 
        self.set_eval()
        errors = []
        pose = torch.Tensor([0.,0.,0.,1]).numpy()
        points = []
        scale = 10
        for batch_idx, inputs in enumerate(tqdm(data_load)):
            for key, ipt in inputs.items(): 
                inputs[key] = ipt.to(self.device) 
            outputs = self.predict_poses(inputs)
            pose = np.matmul(outputs[("cam_T_cam")][0].detach().cpu().numpy(), pose)
            points.append(pose[None])

            points_np = np.concatenate(points,axis=0)
            x = points_np[:, 0]*-1*scale
            z = points_np[:, 2]*scale
            """
            plt.scatter(x,z)
            plt.plot(x, z, '-o')
            if "taj1" in path:
                print('=============1')
                plt.xlim(-50., 50.)
                plt.ylim(0, -100)
            if "taj2" in path:
                print('=============2')
                plt.xlim(-30., 30.)
                plt.ylim(0, -60)
            plt.grid()
            plt.savefig('{}/xz_{:04d}.png'.format(path, batch_idx))
            plt.close()
            """
            plt.style.use('ggplot')
            plt.plot(x[:-1], z[:-1], linewidth=2., color="lightskyblue")
            plt.plot(x[-1], z[-1], 'g*',  linewidth=6., color="dodgerblue")
            if "taj1" in path:
                plt.title("Trajectory of Scene1")
                plt.xlim(-50., 50.)
                plt.ylim(0, -100)
            if "taj2" in path:
                plt.title("Trajectory of Scene2")
                plt.xlim(-30., 30.)
                plt.ylim(0, -60)
            plt.ylabel("Vertical Translation (m)")
            plt.xlabel("Horizontal Translation (m)")
            plt.savefig('{}/xz_{:04d}.png'.format(path, batch_idx))
            plt.close()
        np.savetxt('{}/pose_x.txt'.format(path), x)
        np.savetxt('{}/pose_z.txt'.format(path), z)


    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        pose_inputs = [inputs["color", 0, 0], inputs["color", 1, 0]]
        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
        axisangle, translation = self.models["pose"](pose_inputs)
        outputs[("axisangle")] = axisangle
        outputs[("translation")] = translation
        outputs[("cam_T_cam")] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=False)
        return outputs

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        #writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def load_model(self):
        """Load model(s) from disk"""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
