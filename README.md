# MaST-Pre

## Masked Spatio-Temporal Structure Prediction for Self-supervised Learning on Point Cloud Videos (ICCV 2023)

Visualizations of Reconstruction Results. For each action sample, the ground truth is on the left, and the reconstruction result at 75% masking ratio is on the right.
<br/>
<img src="https://github.com/JohnsonSign/MaST-Pre/blob/main/images/1.gif" width="300">
<img src="https://github.com/JohnsonSign/MaST-Pre/blob/main/images/2.gif" width="300"><br/>
<img src="https://github.com/JohnsonSign/MaST-Pre/blob/main/images/3.gif" width="300">
<img src="https://github.com/JohnsonSign/MaST-Pre/blob/main/images/4.gif" width="300">


## Installation
The code is tested with Python 3.7.12, PyTorch 1.7.1, GCC 9.4.0, and CUDA 10.2.
Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413) and Chamfer_Distance_Loss:
```
cd modules
python setup.py install

cd ./extensions/chamfer_dist
python setup.py install
```

## Related Repositories  
We thank the authors of related repositories:
1. PSTNet: https://github.com/hehefan/Point-Spatio-Temporal-Convolution
2. P4Transformer: https://github.com/hehefan/P4Transformer
3. MAE: https://github.com/facebookresearch/mae
