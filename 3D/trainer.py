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
import open3d
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

class Trainer:
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
        # ----------- TRAINING ----------
        train_dataset_k1 = datasets.DBRAWDataset(
            self.opt.data_path, 'scene1', self.opt.height, self.opt.width, [-1, 0, 1], 4, is_train=True, img_ext='.png')
        self.train_loader_k1 = DataLoader(
            train_dataset_k1, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        train_dataset_k2 = datasets.DBRAWDataset(
            self.opt.data_path, 'scene2', self.opt.height, self.opt.width, [-1, 0, 1], 4, is_train=True, img_ext='.png')
        self.train_loader_k2 = DataLoader(
            train_dataset_k2, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
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

        """
        # ----------- TRAINING ----------
        train_dataset_k = datasets.KITTIRAWDataset(
            "/mnt/nas/kaichen/kitti", train_filenames_k, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext='.jpg')
        self.train_loader_k = DataLoader(
            train_dataset_k, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # ----------- VALIDATION ----------
        val_dataset1 = datasets.KITTIRAWDataset( 
            "/mnt/nas/kaichen/kitti", val_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext=img_ext)
        self.val_loader1 = DataLoader(
            val_dataset1, 1, False, num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=False)

        val_dataset2 = datasets.KITTIRAWDataset( 
            "/mnt/nas/kaichen/kitti", val_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext=img_ext)
        self.val_loader2 = DataLoader(
            val_dataset2, 1, False, num_workers=self.opt.num_workers, 
            pin_memory=True, drop_last=False)
        """
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
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset_k1)+len(train_dataset_k2), len(val_dataset1)+len(val_dataset2)))
        print("There are {:d} training iteration and {:d} validation iteration\n".format(
            len(self.train_loader_k1)+len(self.train_loader_k2), len(self.val_loader1)+len(self.val_loader2)))
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode"""
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        self.init_time = time.time()
        """
        if isinstance(self.opt.load_weights_folder,str):
            self.epoch_start = int(self.opt.load_weights_folder[-1]) + 1
        else:
        """
        self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch
            self.run_epoch(self.train_loader_k1, 'mono_model/{}/train1'.format(self.opt.log_name))
            self.run_epoch(self.train_loader_k2, 'mono_model/{}/train1'.format(self.opt.log_name))
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
            if self.epoch == self.opt.num_epochs - self.epoch_start - 1:
                self.test_epoch(self.val_loader1, 'mono_model/{}/dps1'.format(self.opt.log_name))
                self.test_epoch(self.val_loader2, 'mono_model/{}/dps2'.format(self.opt.log_name))
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def run_epoch(self, data_loader, path):
        if not os.path.exists(path):
            os.makedirs(path) 
        """Run a single epoch of training and validation"""
        #print("Threads: " + str(torch.get_num_threads()))
        print("============> Training{} <============".format(self.epoch))
        self.set_train()
        self.every_epoch_start_time = time.time()
        for batch_idx, inputs in enumerate(tqdm(data_loader)):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            if self.step % 2000 == 0 and self.step != 0:
                self.log_time(self.step, duration, losses["loss"].cpu().data)
            self.step += 1

            gt_height, gt_width = 1080, 1920 
            color = cv2.resize(inputs["color_aug", 0, 0][0].permute(1,2,0).detach().cpu().numpy(), dsize=(gt_width, gt_height))
            cv2.imwrite('{}/img{}.png'.format(path, batch_idx), color*255.)

        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))

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
            vmax = np.percentile(pred_depth, 95)
            #plt.imshow(pred_depth, cmap='magma', vmax=vmax)
            #plt.savefig('{}/depth{}.png'.format(path, batch_idx))
            #plt.axis('off')
            depth_image = disp2rgb(pred_depth)*255.
            cv2.imwrite('{}/depth{}.png'.format(path, batch_idx), depth_image)
            color = cv2.resize(inputs["color", 0, 0][0].permute(1,2,0).detach().cpu().numpy(), dsize=(gt_width, gt_height))
            cv2.imwrite('{}/img{}.png'.format(path, batch_idx), color*255.)
            intrinsic = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            #point = open3d.geometry.create_point_cloud_from_depth_image(pred_depth, intrinsic)
            #o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses"""
        for key, ipt in inputs.items(): # inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device) # put tensor in gpu memory
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}
        # In this setting, we compute the pose to each source frame via a separate forward pass through the pose network.
        # select what features the pose network takes as input
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        # pose_feats is a dict: key: """ keys 0 -1 1 """
        for f_i in self.opt.frame_ids[1:]: # frame_ids = [0,-1,1]
            if f_i != "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) 
            # disp_to_depth function is in layers.py
            outputs[("depth", 0, scale)] = depth
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)], padding_mode="border")
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []
            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(0)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            smooth_loss2 = grad_smooth_loss(norm_disp, H=240)
            loss += 5. * smooth_loss2 / (2 ** scale) # 10, 50, 5
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        total_loss /= self.num_scales
        losses["loss"] = total_loss 
        return losses
    """
    def compute_depth_losses(self, inputs, outputs, losses={}):
        pred_depth = outputs[("depth", 0, 0)]
        gt_depth = inputs['gt_depth'].cpu()[0].numpy()
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = pred_depth.cpu()[0, 0].detach().numpy()
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        depth_errors = compute_errors(gt_depth, pred_depth)
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i]
        return losses
    """
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

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3