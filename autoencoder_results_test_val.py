import numpy as np
import os
from tqdm import tqdm
import os
import datetime
import pandas as pd
import cv2
import math
import statistics
import json
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform

from scipy.ndimage import rotate

from models_pose_est.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
# from val import normalize, pad_width





class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img




class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img




class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),  # Sigmoid for values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# class Autoencoder(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size//3),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size//3, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, input_size),
#             # nn.Softmax()
#             nn.Sigmoid(),  # Sigmoid for values between 0 and 1
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x



def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad



def run_demo(net, image_provider, height_size, cpu, track, smooth, draw):
    # # Initialize the model
    model = Autoencoder(input_size=18, hidden_size=9)
    #
    # # Load the trained weights
    model.load_state_dict(torch.load('simple_autoencoder.pth'))
    model.eval()
    model.to('cuda')

    losses = []

    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    poses_list = []

    detections = 0
    # pbar = tqdm(total=image_provider.max_idx, position=1, leave=True)
    pbar = tqdm(total=image_provider.max_idx)

    for img in image_provider:
        orig_img = img.copy()
        # img = cv2.imread('/home/thodoris/COCO/open_arms.jpg')

        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        if len(pose_entries) > 0:
            detections += 1
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

                ###############
                ## Tried to rotate the predicted keypoints to check if reconstruction loss increases or not
                ## it looks like it increases
                ## if i rotate the input image it increases also
                ## may works as anomaly detector for poses

                ## if uncomment those 2 lines the pose is predicted on normal image and then rotate the pose to see the result
                # rot = rotate(pose_keypoints, 180)
                # pose_keypoints = rot

                ##############

                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            # if track:
            #     track_poses(previous_poses, current_poses, smooth=smooth)
            #     previous_poses = current_poses


            # # Anomaly detection
            list_pose_keypoints = [pose_keypoints]
            distance_matrices = []
            similarity_matrices = []
            for i in range(len(list_pose_keypoints)):
                pairwise_distances = pdist(list_pose_keypoints[i], metric='euclidean')
                distance_matrix = squareform(pairwise_distances)
                sigma = 1
                similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))
                distance_matrices.append(distance_matrix)
                similarity_matrices.append(similarity_matrix)

            input_data = np.stack(similarity_matrices)
            input_tensor = torch.from_numpy(input_data)
            input_tensor = input_tensor.to('cuda')
            input_tensor = input_tensor.float()

            reconstructed_poses = model(input_tensor).detach().cpu().numpy()
            #

            criterion = nn.MSELoss()
            loss = criterion(model(input_tensor), input_tensor)
            # print(loss.item())
            losses.append(loss.item())


            poses_list.append(pose_keypoints)

            if draw:
                # #Get the dimensions of the image
                # (h, w) = img.shape[:2]
                #
                # # Define the center of the image
                # center = (w // 2, h // 2)
                #
                # # Perform the rotation
                # M = cv2.getRotationMatrix2D(center, 90, 1.0)
                # img = cv2.warpAffine(img, M, (w, h))
                # #DRAW
                for pose in current_poses:
                    pose.draw(img)
                img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
                for pose in current_poses:
                    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                    position = (10, img.shape[0] - 10)
                    cv2.putText(img, 'loss is: '+str(round(loss.item(), 5)), position,
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                    if track:
                        cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
                key = cv2.waitKey(0)
                if key == 27:  # esc
                    return
                elif key == 112:  # 'p'
                    if delay == 1:
                        delay = 0
                    else:
                        delay = 1
            pbar.update()

    return poses_list, losses, detections



checkpoint_path = '/home/thodoris/lightweight-human-pose-estimation_pytorch/checkpoint/checkpoint_iter_370000.pth'
height_size = 256
cpu = False
track = False
smooth = False
draw = True
# imgs_folder = '/home/thodoris/Desktop/testing/'
# list_images = []
# for file in os.listdir(imgs_folder):
#     list_images.append(imgs_folder+file)

# list_images = ['/home/thodoris/Downloads/gazebo_test_edit.jpg']
# list_images = ['./camera_rear.jpg']

list_images = ['./test_images/open_arms.jpg', './test_images/open_arms_180.jpg', './test_images/open_arms_edit.jpg']

# video = 'path to video'
# video = 0

# eval on rotation imgs
# angle = 10


net = PoseEstimationWithMobileNet()
checkpoint = torch.load(checkpoint_path, map_location='cuda')
load_state(net, checkpoint)




frame_provider = ImageReader(list_images)#



pose_keypoints, losses, detections = run_demo(net, frame_provider, height_size, cpu, track, smooth, draw)


print("loss:", losses)

## Print some statistics

# avg_loss = statistics.mean(losses)
# std_loss = statistics.stdev(losses)
# median_loss = statistics.median(losses)
# variance_loss = statistics.variance(losses)

# print(f"\nAvg loss  ", avg_loss)
# print(f"\nStdev loss: ", std_loss)
# print(f"\nVariance loss: ", variance_loss)
# print(f"\nMedian loss: ", median_loss)





