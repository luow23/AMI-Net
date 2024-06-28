# [AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization (IEEE TASE 2024)](https://ieeexplore.ieee.org/document/10445116)
PyTorch implementation and for TASE2024 paper, AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization.  
![这是图片](AMI-Net-framework.png)  
# Download Weights of MVTec AD Dataset
| Class      | Pre-trained Checkpoint |  Metric (I-AUROC,P-AUROC,I-AP,P-AP)    |
|------------|-------------------------|--------|
| Bottle   | [download](https://drive.google.com/drive/folders/1hReDmbzSDeKSjUgKVfNvwS7eiLuZvqlO?usp=drive_link) | (1.0, 0.988, 1.0, 0.792) |
| Cable   | [download](https://drive.google.com/drive/folders/1BzQ6dJQoGGnh672Z_1nULDXDfu4MjnlR?usp=drive_link) | (0.996, 0.986, 0.998, 0.685) |
| Capsule   | [download](https://drive.google.com/drive/folders/19E5Sb6v4L_rCL-8wprVJcPSPj_-u3D6g?usp=drive_link) | (0.984, 0.989, 0.997, 0.45) |
| Carpet   | [download](https://drive.google.com/drive/folders/1H4WLy7Qx_8-wcUmPDTJoBiC-fzBHee4w?usp=drive_link) | (0.998, 0.993, 0.999, 0.69) |
| Grid   | [download](https://drive.google.com/drive/folders/1iAK-jcxTzMXxJGcJkubbwCQ4I8L6rsTR?usp=drive_link) | (0.999, 0.989, 1.0, 0.378) |
| Hazelnut   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (1.0, 0.986, 1.0, 0.567) |
| Leather   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (1.0, 0.994, 1.0, 0.486 |
| Metal nut   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.995, 0.966, 0.999, 0.672) |
| Pill   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.966, 0.983, 0.994, 0.697) |
| Screw  | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.978, 0.994, 0.993, 0.369) |
| Tile   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.999, 0.962, 1.0, 0.552) |
| Toothbrush   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.958, 0.989, 0.984, 0.519) |
| Transistor   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (1.0, 0.981, 1.0, 0.771) |
| Wood   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.993, 0.953, 0.998, 0.478) |
| Zipper   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | (0.986, 0.985, 0.996, 0.53) |


# Download Datasets
Please download MVTecAD dataset from [MVTecAD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) and BTAD dataset from [BTAD dataset](https://www.beantech.it/).
# Installation
[timm==0.3.2](https://github.com/huggingface/pytorch-image-models)     
[pytoch==1.8.1](https://pytorch.org/)
# Citation
If you find this repository useful, please consider citing our work:  
```
@article{luo2024ami,    
  title={AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization},  
  author={Luo, Wei and Yao, Haiming and Yu, Wenyong and Li, Zhengyong},  
  journal={IEEE Transactions on Automation Science and Engineering},  
  year={2024},  
  publisher={IEEE}  
}
```
