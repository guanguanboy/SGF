



# Segmentation Guided Low-light Image Enhancement

## News
- **Nov 23, 2023:** Codes, datasets, and pre-trained models will be released!

## Installation

1. Clone our repository
```
Download our code.
cd SGF
```

2. Make conda environment
```
conda create -n pytorch111 python=3.7
conda activate pytorch111
```

3. Install dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr
```
python setup_basicsr.py develop --no_cuda_ext
```

## Prepare Dataset
Download the following datasets:

LOL-v1 [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

LOL-v2 [Baidu Disk](https://pan.baidu.com/s/1X4HykuVL_1WyB3LWJJhBQg?pwd=cyh2) (code: `cyh2`), [Google Drive](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)

SID [Baidu Disk](https://pan.baidu.com/share/init?surl=HRr-5LJO0V0CWqtoctQp9w) (code: `gplv`), [Google Drive](https://drive.google.com/drive/folders/1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR?usp=share_link&pli=1)


## Training
Training instructions on different datasets are listed as follows. 

### LOL-v1

training code:

```
./train_sam.sh Enhancement/Options/Enhancement_SGF_Lolv1.yml
```

### LOL-v2-real

training code:

```
 ./train_sam.sh Enhancement/Options/Enhancement_SGF_Lolv2_real.yml
```

### LOL-v2-synthetic

training code:

```
./train_sam.sh Enhancement/Options/Enhancement_SGF_Lolv2_synthetic.yml
```

### SID

training code:

```
./train_sam.sh Enhancement/Options/Enhancement_SGF_SID.yml
```


## Evaluation

Fisrt download the pretained model from [Google Drive](https://drive.google.com/drive/folders/1N_qeQuP4EZJ3lBs0mG8S9oT8qvXpY0YE?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/13L-EROAtlOGNUrBJFehQsQ)（code: `pacs`）and put them in the root directory.

Evaluation instructions on different datasets are listed as follows. 

### LOL-v1

testing code:

```
python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGF_Lolv1.yml --weights pretrained_models/LOLv1/net_g_latest.pth --dataset LOLv1_edge
```

### LOL-v2-real

testing code:

```
python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGF_Lolv2_real.yml --weights pretrained_models/LOLv2_real/net_g_latest.pth --dataset LOLv2
```

### LOL-v2-synthetic

testing code:

```
python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGF_Lolv2_synthetic.yml --weights pretrained_models/LOLv2_synthetic/net_g_latest.pth --dataset LOLv2_synthetic
```

### SID

testing code:

```
python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGF_SID.yml --weights pretrained_models/SID/net_g_latest.pth --dataset SID_SAM
```

## Results
Experiments are performed for low-light image enhancement or four benchmark dataset.


## Citation
If you use our code, please consider citing our paper:




## Contact
Should you have any question, please contact liguanlin1229@gmail.com


**Acknowledgment:** This code is based on the [BasicSR](https://github.com/xinntao/BasicSR) toolbox. 

## Our Related Works
- Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022. [Paper](https://arxiv.org/abs/2111.09881) | [Code](https://github.com/swz30/Restormer)
- Learning Enriched Features for Real Image Restoration and Enhancement, ECCV 2020. [Paper](https://arxiv.org/abs/2003.06792) | [Code](https://github.com/swz30/MIRNet)

