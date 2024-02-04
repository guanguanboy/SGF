import os.path as osp
import torch
import torch.utils.data as data
import basicsr.data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools


class Dataset_SIDImageSemantic(data.Dataset):
    def __init__(self, opt):
        super(Dataset_SIDImageSemantic, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.semantic_folder = opt['dataroot_semantic']
        self.io_backend_opt = opt['io_backend']
        self.data_type = opt['io_backend']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'path_SM': [],
                          'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT, self.imgs_SM = {}, {}, {}

        subfolders_LQ_origin = util.glob_file_list(self.LQ_root)
        subfolders_GT_origin = util.glob_file_list(self.GT_root)
        subfolders_semantic_origin = util.glob_file_list(self.semantic_folder)

        subfolders_LQ = []
        subfolders_GT = []
        subfolders_SM = []

        if self.opt['phase'] == 'train':
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
                    subfolders_SM.append(subfolders_semantic_origin[mm])                                                        
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
                    subfolders_SM.append(subfolders_semantic_origin[mm])

        for subfolder_LQ, subfolder_GT, subfolders_SM in zip(subfolders_LQ, subfolders_GT,subfolders_SM):
            # for frames in each video:
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)
            img_paths_SM = util.glob_file_list(subfolders_SM)


            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['path_SM'].extend(img_paths_SM)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT
                self.imgs_SM[subfolder_name] = img_paths_SM

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_LQ_path = [img_LQ_path]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]

        img_SM_path = self.imgs_SM[folder][idx]
        img_SM_path = [img_SM_path]

        if self.opt['phase'] == 'train':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            img_LQ_l.append(img_SM)

            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]
            img_SM = rlt[2]

        elif self.opt['phase'] == 'test':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]
        else:
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]
        # img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
        # img_nf = cv2.blur(img_nf, (5, 5))
        # img_nf = img_nf * 1.0 / 255.0
        # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        return {
            'lq': img_LQ,
            'gt': img_GT,
            'semantic' : img_SM,
            # 'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])

class Dataset_SIDSamGrayIllImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_SIDSamGrayIllImage, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.semantic_folder = opt['dataroot_semantic']
        self.io_backend_opt = opt['io_backend']
        self.data_type = opt['io_backend']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'path_SM': [],
                          'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT, self.imgs_SM = {}, {}, {}

        subfolders_LQ_origin = util.glob_file_list(self.LQ_root)
        subfolders_GT_origin = util.glob_file_list(self.GT_root)
        subfolders_semantic_origin = util.glob_file_list(self.semantic_folder)

        subfolders_LQ = []
        subfolders_GT = []
        subfolders_SM = []

        if self.opt['phase'] == 'train':
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
                    subfolders_SM.append(subfolders_semantic_origin[mm])
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
                    subfolders_SM.append(subfolders_semantic_origin[mm])

        for subfolder_LQ, subfolder_GT, subfolders_SM in zip(subfolders_LQ, subfolders_GT,subfolders_SM):
            # for frames in each video:
            subfolder_name = osp.basename(subfolder_LQ)

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)
            img_paths_SM = util.glob_file_list(subfolders_SM)


            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['path_SM'].extend(img_paths_SM)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT
                self.imgs_SM[subfolder_name] = img_paths_SM

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_LQ_path = [img_LQ_path]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]

        img_SM_path = self.imgs_SM[folder][idx]
        img_SM_path = [img_SM_path]

        if self.opt['phase'] == 'train':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            img_LQ_l.append(img_SM)

            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]
            img_SM = rlt[2]

        elif self.opt['phase'] == 'test':
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]
        else:
            img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])
            img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]
            img_SM = util.read_img_seq_gray(img_SM_path, self.opt['train_size'])
            img_SM = img_SM[0]
        # img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
        # img_nf = cv2.blur(img_nf, (5, 5))
        # img_nf = img_nf * 1.0 / 255.0
        # img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        #gray scale illumination map
        r,g,b = img_LQ[0]+1, img_LQ[1]+1, img_LQ[2]+1
        img_gray_illum = 1. - (0.299*r+0.587*g+0.114*b)/2.
        img_gray_illum = torch.unsqueeze(img_gray_illum, 0)

        semantic = torch.cat([img_SM, img_gray_illum],dim=0)

        return {
            'lq': img_LQ,
            'gt': img_GT,
            'semantic' : semantic,
            # 'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])