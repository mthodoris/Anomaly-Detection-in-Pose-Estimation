import os
import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

img_folder = '/home/thodoris/Desktop/pose_dataset/val_single_poses/'

angles = [i for i in range(170, 181, 10)]
pbar = tqdm(total=len(angles))
# pbar2 = tqdm(total=len(os.listdir(img_folder)))

for angle in angles:
    os.makedirs(f'/home/thodoris/Desktop/pose_dataset/rotated_{angle}_imgs/', exist_ok=True)
    rot_dir = f'/home/thodoris/Desktop/pose_dataset/rotated_{angle}_imgs/'
    pbar2 = tqdm(total=len(os.listdir(img_folder)))
    for img_name in os.listdir(img_folder):
        img = cv2.imread(img_folder+img_name)
        rotated_img = ndimage.rotate(img, angle)
        # cv2.imwrite(f'/home/thodoris/Desktop/pose_dataset/rotated_{angle}_imgs/{img_name}', rotated_img)
        cv2.imwrite(rot_dir + img_name, rotated_img)
        pbar2.update()
    pbar.update()



print('end')