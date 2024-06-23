from models.model_MAE import *
from models.networks import *

from torch import nn
import random






class AMAE(nn.Module):
    def __init__(self, opt):
        super(AMAE, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model =Adpative_MAE_k_center(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma, clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = Adpative_MAE_k_center(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, center_num=opt.center_num, sigma=opt.sigma,  clu_depth=opt.clu_lay_num)

    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature, cos_sim, bin_mask  = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss, cos_sim, bin_mask

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        dis_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        dis_map = nn.functional.interpolate(dis_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dis_map = dis_map.clone().squeeze(0).cpu().detach().numpy()

        dir_map = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        dir_map = dir_map.reshape(batch_size, 1, 64, 64)
        dir_map = nn.functional.interpolate(dir_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dir_map = dir_map.clone().squeeze(0).cpu().detach().numpy()
        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return dis_map, dir_map


