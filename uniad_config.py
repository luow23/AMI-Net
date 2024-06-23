import torch

model_name_list = [ 'AMAE']

class DefaultConfig(object):
    class_name = 'bottle'
    data_root = r'E:\GAM-Net2\data\mvtec_anomaly_detection'
    device = torch.device('cuda:0')
    model_name = model_name_list[0]
    batch_size = 16
    iter = 0
    niter = 300
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nc = 3
    isTrain = True
    backbone_name = 'WideResnet50'
    resume =r''
    sigma = 0.5
    center_num = 8
    k = 4
    in_lay_num = 8
    clu_lay_num = 1


if __name__ == '__main__':
    opt = DefaultConfig()
    opt.trai = 1
    print(opt.trai)