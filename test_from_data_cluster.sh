echo "Begin init conda..."
. /mnt/data/optimal/zhaobin_gl/miniconda3/etc/profile.d/conda.sh
which conda

echo "Begin activating conda env..."
conda activate base

echo "Begin training!"
cd /mnt/data/optimal/zhaobin_gl/Codes/SAM_MIRNetv2
python setup.py develop --no_cuda_ext

#python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGIENet_gray_illlum_drconve_Lolv2_synthetic_w_sam.yml --weights experiments/Enhancement_SGIENet_lolv2_synthetic_gray_drconv_illum_sam_1021/models/net_g_latest.pth --dataset Dataset_PairedWithGrayIllumImage

#python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_NAFRestormer_Lolv1_full_w_sam.yml --weights experiments/Enhancement_NAFRestormer_Lolv1_full_w_sam_1021/models/net_g_latest.pth --dataset Dataset_PairedWithGrayIllumImage

python3 Enhancement/test_from_dataset.py --opt Enhancement/Options/Enhancement_SGIENet_gray_illlum_drconve_Lol_w_sam_sid_cluster.yml --weights experiments/Enhancement_SGIENet_lol_gray_drconv_illum_sam_1020/models/net_g_latest.pth --dataset SID