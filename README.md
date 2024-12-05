# [AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization (IEEE TASE 2024)](https://ieeexplore.ieee.org/document/10445116)
PyTorch implementation and for TASE2024 paper, AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization.  
![这是图片](AMI-Net-framework.png)  
# Download Weights of MVTec AD Dataset
| Class      | Pre-trained Checkpoint |  Metric (I-AUROC,P-AUROC,I-AP,P-AP)    |
|------------|-------------------------|--------|
| Bottle   | [download](https://drive.google.com/file/d/1B0vZxRfQ21pG17K3iLUt7ADnziFHAKzK/view?usp=sharing) | (1.0, 0.988, 1.0, 0.792) |
| Cable   | [download](https://drive.google.com/file/d/1YgB0raWusFhe1albFVndJU-TGDwb8zb2/view?usp=sharing) | (0.996, 0.986, 0.998, 0.685) |
| Capsule   | [download](https://drive.google.com/file/d/1KKIF4DpZPPVqlUWErcSpAzpTBpY7DcTJ/view?usp=sharing) | (0.984, 0.989, 0.997, 0.45) |
| Carpet   | [download](https://drive.google.com/file/d/1Svp2NKQvoilXWBD1nxfCRlX9weuZWVVq/view?usp=sharing) | (0.998, 0.993, 0.999, 0.69) |
| Grid   | [download](https://drive.google.com/file/d/1DZZx6NaacOuwbADX1d4T-h_gg9GWAGql/view?usp=sharing) | (0.999, 0.989, 1.0, 0.378) |
| Hazelnut   | [download](https://drive.google.com/file/d/1CWVN9LhdrP3qfbxMa8g0-00a4uV29j9Y/view?usp=sharing) | (1.0, 0.986, 1.0, 0.567) |
| Leather   | [download](https://drive.google.com/file/d/1Ps1x86dhrS1rOU9hXpPFBfiNX1T6HZHh/view?usp=sharing) | (1.0, 0.994, 1.0, 0.486 |
| Metal nut   | [download](https://drive.google.com/file/d/150DJN0tgID_wQYGgasfzIbh_utNw91uC/view?usp=sharing) | (0.995, 0.966, 0.999, 0.672) |
| Pill   | [download](https://drive.google.com/file/d/1odETkF0e-Yp_ovMlkPtMhZD9gFjV2Q8O/view?usp=sharing) | (0.966, 0.983, 0.994, 0.697) |
| Screw  | [download](https://drive.google.com/file/d/1DxCl2uiMZvUFEVd0DItAcznhfpSOLU26/view?usp=sharing) | (0.978, 0.994, 0.993, 0.369) |
| Tile   | [download](https://drive.google.com/file/d/1GGIkQ7pKQYB2wcQZuV7gho2NNEGuMPuy/view?usp=sharing) | (0.999, 0.962, 1.0, 0.552) |
| Toothbrush   | [download](https://drive.google.com/file/d/1TZ3cYMmMrAU3S265ETNqCH4Mfj_zrA9G/view?usp=sharing) | (0.958, 0.989, 0.984, 0.519) |
| Transistor   | [download](https://drive.google.com/file/d/1okoeOQ3Bs_CyonjTdswHHte1qls1_JpA/view?usp=sharing) | (1.0, 0.981, 1.0, 0.771) |
| Wood   | [download](https://drive.google.com/file/d/1up1GdL7Gtqt2Tuq_J5r-yym0EUY7q1eF/view?usp=sharing) | (0.993, 0.953, 0.998, 0.478) |
| Zipper   | [download](https://drive.google.com/file/d/1zYu7Rzd7QCItTltS1rHr0jjI_vzn0BTq/view?usp=sharing) | (0.986, 0.985, 0.996, 0.53) |


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
