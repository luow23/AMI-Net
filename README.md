# [AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization (IEEE TASE 2024)](https://ieeexplore.ieee.org/document/10445116)
PyTorch implementation and for TASE2024 paper, AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization.  
![这是图片](AMI-Net-framework.png)  
# Download Weights of MVTec AD Dataset
| Class      | Pre-trained Checkpoint |  Metric (I-AUROC,P-AUROC,I-AP,P-AP)    |
|------------|-------------------------|--------|
| Bottle   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=drive_link) | 8cad7c |
| ViT-Large  | [download](https://github.com/your-username/your-repo/releases/download/v1.0/ViT-Large.pth) | b8b06e |
| ViT-Huge   | [download](https://github.com/your-username/your-repo/releases/download/v1.0/ViT-Huge.pth) | 9bdbb0 |

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
