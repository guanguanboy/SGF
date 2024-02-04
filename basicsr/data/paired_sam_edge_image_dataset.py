from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    tripled_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,tripled_paths_json_from_folder)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation,paired_random_crop_semantic
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import os
import json
from pycocotools import mask as mask_utils
import random
seed_value = 42  # 随机种子值

random.seed(seed_value)

class Dataset_PairedSamEdgeImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedSamEdgeImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder, self.semantic_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_semantic']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'


        self.paths = tripled_paths_from_folder(
            [self.lq_folder, self.gt_folder,self.semantic_folder], ['lq', 'gt', 'semantic'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_gt_for_edge = cv2.imread(gt_path)

        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        semantic_path = self.paths[index]['semantic_path']
        img_bytes = self.file_client.get(semantic_path, 'semantic')
        try:
            img_semantic = imfrombytes(img_bytes, flag='grayscale', float32=True)
            img_semantic = np.expand_dims(img_semantic, axis=-1)

        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_lq, img_semantic,img_gt = paired_random_crop_semantic(img_lq, img_semantic, img_gt, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq, img_semantic = random_augmentation(img_gt, img_lq, img_semantic)

        #get edge
        im_gt_gray = cv2.cvtColor(img_gt_for_edge, cv2.COLOR_BGR2GRAY)
        sketch = cv2.GaussianBlur(im_gt_gray, (3, 3), 0)

        v = np.median(sketch)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        sketch = cv2.Canny(sketch, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sketch = cv2.dilate(sketch, kernel)

        sketch = np.expand_dims(sketch, axis=-1)
        sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
        assert len(np.unique(sketch)) == 2            
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        img_semantic = img2tensor(img_semantic, bgr2rgb=False, float32=True)
        img_gt_edge = torch.from_numpy(sketch).permute(2, 0, 1)
        img_gt_edge = img_gt_edge[0:1, :, :]
        img_gt_edge = img_gt_edge.long()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'semantic' : img_semantic,
            'semantic_path' : semantic_path,
            'gt_edge' : img_gt_edge
        }

    def __len__(self):
        return len(self.paths)
