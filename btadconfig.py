import torch

model_name_list = ['VIT_Dir', 'AMAE', 'AMAE_kaiming','AMAE_re_add_in']

class DefaultConfig(object):
    class_name = 'bottle'
    data_root = r'E:\GAM-Net2\data\BTech_Dataset_transformed'
    device = torch.device('cuda:0')
    model_name = model_name_list[1]
    batch_size = 8
    iter = 0
    niter = 300
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nc = 3
    isTrain = True
    backbone_name = 'WideResnet50'
    # referenc_img_file = f''
    resume =r''
    # mask_ratio = 0.5
    sigma = 0.5
    center_num = 8
    k = 4
    in_lay_num = 8
    clu_lay_num = 1


if __name__ == '__main__':
    opt = DefaultConfig()
    opt.trai = 1
    print(opt.trai)