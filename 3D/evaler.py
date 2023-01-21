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

MIN_DEPTH = 0.1
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

def get_pixelgrid(h, w):
    grid_h = torch.linspace(w - 1, 0., w).view(1, 1, w).expand(1, h, w)
    grid_v = torch.linspace(h - 1, 0., h).view(1, h, 1).expand(1, h, w)    
    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=0)
    return pixelgrid

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
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
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
            self.opt.data_path, 'scene1', self.opt.height, self.opt.width, [0], 4, is_train=False, img_ext='.png')
        self.val_loader1 = DataLoader(
            val_dataset1, 1, False, num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=False)
        val_dataset2 = datasets.DBRAWDataset(
            self.opt.data_path, 'scene2', self.opt.height, self.opt.width, [0], 4, is_train=False, img_ext='.png')
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
        self.test_epoch(self.val_loader1, 'mono_model/{}/depth1'.format(self.opt.log_name))
        self.test_epoch(self.val_loader2, 'mono_model/{}/depth2'.format(self.opt.log_name))
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def test_epoch(self, data_load, path):
        print("============> Validation{} <============".format(self.epoch))
        if not os.path.exists(path):
            os.makedirs(path) 
        self.set_eval()
        errors = []
        for batch_idx, inputs in enumerate(tqdm(data_load)):
            for key, ipt in inputs.items(): 
                inputs[key] = ipt.to(self.device) 
            features = self.models["encoder"](inputs["color", 0, 0])
            outputs = self.models["depth"](features)
            disp = outputs[("disp", 0)]
            pred_disp_0, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) 
            pred_disp = pred_disp_0.detach().cpu()[0, 0].numpy()
            gt_height, gt_width = 1080, 1920 
            pred_disp = cv2.resize(pred_disp, dsize=(gt_width, gt_height))
            pred_depth = 1 / pred_disp
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            mask = inputs["mask"].cpu().numpy()[0,:,:,0]
            mask = mask == 0
            print(np.max(pred_depth))
            pred_depth[mask] = 15      
            MASK = 1-mask
            """
            h, w = pred_depth.shape
            hp = get_pixelgrid(h, w)
            points = (hp.numpy()*pred_depth[None]).reshape(3, -1)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points[0], points[1], points[2], marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.savefig('xz_11.png')
            plt.close()
            """
            depth_image = disp2rgb(pred_depth)*255.
            #with open('{}/depth{:04d}.npy'.format(path, batch_idx), 'wb') as f:
            #    np.save(f, np.half(pred_depth))
            #cv2.imwrite('{}/depth{:04d}.png'.format(path, batch_idx), depth_image)
            #torch.save(torch.from_numpy(pred_depth).half(), '{}/depth{:04d}.pt'.format(path, batch_idx))
            vmax = np.percentile(pred_depth, 75)
            plt.imsave('{}/depth{:04d}.png'.format(path, batch_idx), pred_depth, cmap='magma', vmax=vmax)
            plt.close()

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

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

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
