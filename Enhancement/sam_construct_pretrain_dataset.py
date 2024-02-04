# This script batch segments pictures in <source> directory using SAM（Segment Anything）.
# And save the semantic pictures in <source_semantic> for colored versions,
# and <source_semanticB> for binary versions

# Author: Qihan Zhao

# Prerequisite:
# 1. download this script in directory `SAM`
# 2. download SAM weights in directory `SAM`
# 3. add <source> file in directory `SAM`

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--source_dir', type=str, default='Datasets/our485/low', help='directory of data to be segmented')
args = parser.parse_args()




sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

gt_dir = 'Datasets/our485/high'
import os
sourcedir = args.source_dir
for i, filename in enumerate(os.listdir(sourcedir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print(f'{i}th pic: {filename}')
        # read image 
        img_path = os.path.join(sourcedir, filename)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_gt_path = os.path.join(gt_dir, filename)
        image_gt = cv2.imread(img_gt_path)

        # segment
        masks = mask_generator.generate(image_rgb)
        
        # save semantic
        gray_img = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 1), dtype=np.int8)
        #img[:,:,3] = 0
            
        # Binary
        folder_name = 'sam_masks'
        os.makedirs(folder_name, exist_ok=True)

        low_folder_name = 'low_sam'
        high_folder_name = 'high_sam'
        os.makedirs(low_folder_name, exist_ok=True)
        os.makedirs(high_folder_name, exist_ok=True)

        for i, mask in enumerate(masks):
            mask_file_name = f'{os.path.splitext(filename)[0]}_sam_{i}.png'
            mask_save_path = os.path.join(folder_name, mask_file_name)
            cv2.imwrite(mask_save_path, np.uint8(mask["segmentation"]) * 255)

            low_file_name =  f'{os.path.splitext(filename)[0]}_sam_{i}.jpg'
            low_save_path = os.path.join(low_folder_name, low_file_name)
            cv2.imwrite(low_save_path, image)

            high_file_name =  f'{os.path.splitext(filename)[0]}_sam_{i}.jpg'
            high_save_path = os.path.join(high_folder_name, high_file_name)
            cv2.imwrite(high_save_path, image_gt)


        