from config import DefaultConfig
import os
from torch import optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import cv2
from models.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.ndimage import gaussian_filter
from datasets.dataset import denormalize
from models.AMI_Net_model import *
from sklearn import manifold


class Model(object):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = eval(opt.model_name)(opt)
        self.device = opt.device

        self.class_name = opt.class_name
        self.trainloader = opt.trainloader
        self.testloader = opt.testloader
        self.loss_scaler = NativeScaler()
        if self.opt.resume != "":
            print('\nload pre-trained networks')
            self.opt.iter = \
            torch.load(os.path.join(self.opt.resume, self.opt.class_name, f'{opt.model_name}.pth'))[
                'epoch']
            print(self.opt.iter)
            self.model.load_state_dict(torch.load(
                os.path.join(self.opt.resume, self.opt.class_name, f'{opt.model_name}.pth'))[
                                           'state_dict'], strict=False)
            print('\ndone.\n')

        if self.opt.isTrain:
            self.model.Roncon_model.train()
            self.optimizer_g = optim.AdamW(self.model.Roncon_model.parameters(), lr=opt.lr, betas=(0.9, 0.95))

        self.save_root = f"./result_test/{opt.model_name}_{opt.backbone_name}_k={opt.k}_cnum={opt.center_num}_sigma={opt.sigma}_inpaint_num={opt. in_lay_num}_clu_num={opt.clu_lay_num}/"
        # os.makedirs(os.path.join(self.save_root, "weight"), exist_ok=True)
        self.ckpt_root = os.path.join(self.save_root, "weight/{}".format(self.class_name))
        self.vis_root = os.path.join(self.save_root, "img/{}".format(self.class_name))
    def get_max(self, tensor):
        a_1, _ = torch.max(tensor, dim=1, keepdim=True)
        a_2, _ = torch.max(a_1, dim=2, keepdim=True)
        a_3, _ = torch.max(a_2, dim=3, keepdim=True)
        return a_3

    def cal_auc(self, score_list, score_map_list, test_y_list, test_mask_list):
        flatten_y_list = np.array(test_y_list).ravel()
        flatten_score_list = np.array(score_list).ravel()
        image_level_ROCAUC = roc_auc_score(flatten_y_list, flatten_score_list)
        image_level_AP = average_precision_score(flatten_y_list, flatten_score_list)

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list)
        pixel_level_AP = average_precision_score(flatten_mask_list, flatten_score_map_list)
        # pro_auc_score = 0
        # pro_auc_score = cal_pro_metric_new(test_mask_list, score_map_list, fpr_thresh=0.3)
        return round(image_level_ROCAUC, 3), round(pixel_level_ROCAUC, 3), round(image_level_AP, 3), round(pixel_level_AP, 3)
        # return  image_level_ROCAUC, pixel_level_ROCAUC

    def F1_score(self, score_map_list, test_mask_list):
        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        F1_score = f1_score(flatten_mask_list, flatten_score_map_list)
        return F1_score

    def filter(self, pred_mask):
        pred_mask_my = np.squeeze(np.squeeze(pred_mask, 0), 0)
        pred_mask_my = cv2.medianBlur(np.uint8(pred_mask_my * 255), 7)
        mean = np.mean(pred_mask_my)
        std = np.std(pred_mask_my)
        _ , binary_pred_mask = cv2.threshold(pred_mask_my, mean+2.75*std, 255, type=cv2.THRESH_BINARY)
        binary_pred_mask = np.uint8(binary_pred_mask/255)
        pred_mask_my = np.expand_dims(np.expand_dims(pred_mask_my, 0), 0)
        binary_pred_mask = np.expand_dims(np.expand_dims(binary_pred_mask, 0), 0)
        return pred_mask_my, binary_pred_mask


    # def thresholding(self, pred_mask_my):
    #     np_img

        # return
    def feature_map_vis(self, feature_map_list):
        feature_map_list = [torch.mean(i.clone(), dim=1).squeeze(0).cpu().detach().numpy() for i in feature_map_list]
        return feature_map_list



    def test(self):
        test_y_list = []
        test_mask_list = []
        score_list = []
        score_map_list = []

        for idx, (x, y, mask, name) in enumerate(tqdm(self.testloader, ncols=80)):
            test_y_list.extend(y.detach().cpu().numpy())
            test_mask_list.extend(mask.detach().cpu().numpy())
            self.model.eval()
            self.model.to(self.device)
            x = x.to(self.device)
            mask = mask.to(self.device)
            mask_cpu = mask.cpu().detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            # ref_x = get_pos_sample(self.opt.referenc_img_file, self.device, 1)
            deep_feature, ref_feature, recon_feature, _, cos_sim, bin_mask = self.model(x, 'test')
            cos_sim = cos_sim.view((1, int(64/opt.k),  int(64/opt.k)))
            cos_sim_cpu = cos_sim.cpu().detach().numpy().transpose((1, 2, 0))
            bin_mask = bin_mask.view((1,  int(64/opt.k),  int(64/opt.k)))
            bin_mask_cpu = bin_mask.cpu().detach().numpy().transpose((1, 2, 0))
            feature_map_vis_list = self.feature_map_vis([deep_feature, ref_feature, recon_feature])
            dis_amap, dir_amap = self.model.a_map(deep_feature, recon_feature)
            dis_amap = gaussian_filter(dis_amap, sigma=4)
            dir_amap = gaussian_filter(dir_amap, sigma=4)
            # ssim_amap = gaussian_filter(ssim_amap, sigma=4)
            # print(type(name0]))
            name_list = name[0].split(r'!')
            # print(name_list)
            category, img_name = name_list[-2], name_list[-1]
            amap = dir_amap * dis_amap
            # amap = dir_amap + dis_amap
            self.vis_img(
                [x, *feature_map_vis_list[1:], cos_sim_cpu, bin_mask_cpu, dis_amap, dir_amap, amap, amap, mask_cpu],
                os.path.join(self.vis_root, category), img_name)


            score_list.extend(np.array(np.std(amap)).reshape(1))
            score_map_list.extend(amap.reshape((1, 1, 256, 256)))


        image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP= self.cal_auc(score_list, score_map_list, test_y_list, test_mask_list)
        # F1_score = self.F1_score(F1_score_map_list, test_mask_list)
        print('image_auc_roc: {} '.format(image_level_ROCAUC),
              'pixel_auc_roc: {} '.format(pixel_level_ROCAUC),
              'image_AP: {}'.format(image_level_AP),
              'pixel_AP: {}'.format(pixel_level_AP)
             )
        # class_rocauc[self.opt.class_name] = (image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP)
        return image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP

    def vis_img(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.figure()
        plt.subplot(251)
        plt.imshow(cv2_input)
        plt.axis('off')
        plt.subplot(252)
        plt.imshow(img_list[1])
        plt.axis('off')
        plt.subplot(253)
        plt.imshow(img_list[2])
        plt.axis('off')
        plt.subplot(254)
        plt.imshow(img_list[3], cmap='jet')
        plt.axis('off')
        plt.subplot(255)
        plt.imshow(img_list[4], cmap='gray')
        plt.axis('off')
        plt.grid('on')
        plt.subplot(256)
        plt.imshow(img_list[5], cmap='jet')
        plt.axis('off')
        plt.subplot(257)
        plt.imshow(img_list[6],cmap='jet')
        plt.axis('off')
        plt.subplot(258)
        plt.imshow(img_list[7],cmap='jet')
        plt.axis('off')
        plt.subplot(259)
        plt.imshow(img_list[8], cmap='jet')
        plt.axis('off')
        plt.subplot(2,5,10)
        plt.imshow(img_list[9])
        plt.axis('off')
        plt.savefig(os.path.join(save_root, idx_name))
        plt.close()


    def tensor_to_np_cpu(self, tensor):
        x_cpu = tensor.squeeze(0).data.cpu().numpy()
        x_cpu = np.transpose(x_cpu, (1, 2, 0))
        return x_cpu

    def check(self, img):
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 3:
            return img
        elif img.shape[2] == 1:
            return img.reshape(img.shape[0], img.shape[1])

MVTec_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# MVTec_CLASS_NAMES = ['metal_nut', 'pill', 'screw',
#                      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# MVTec_CLASS_NAMES = ['screw',
#                      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class_rocauc = {
                'bottle':(0, 0, 0, 0),
                'cable':(0, 0, 0, 0),
                'capsule':(0, 0, 0, 0),
                'carpet':(0, 0, 0, 0),
                'grid':(0, 0, 0, 0),
                'hazelnut':(0, 0, 0, 0),
                'leather':(0, 0, 0, 0),
                'metal_nut':(0, 0, 0, 0),
                'pill':(0, 0, 0, 0),
                'screw':(0, 0, 0, 0),
                'tile':(0, 0, 0, 0),
                'toothbrush':(0, 0, 0, 0),
                'transistor':(0, 0, 0, 0),
                'wood':(0, 0, 0, 0),
                'zipper':(0, 0, 0, 0)}
# MVTec_CLASS_NAMES = [ 'carpet', 'grid',
#                       'leather', 'tile', 'wood']
#
# class_rocauc = {
#                 'carpet':(0, 0, 0, 0),
#                 'grid':(0, 0, 0, 0),
#                 'leather':(0, 0, 0, 0),
#                 'tile':(0, 0, 0, 0),
#                 'wood':(0, 0, 0, 0)
#                }

if __name__ == '__main__':
    opt = DefaultConfig()
    from datasets.dataset import MVTecDataset
    from torch.utils.data import DataLoader
    for classname in MVTec_CLASS_NAMES:
        opt.class_name = classname
        # opt.class_name = 'capsule'
        # opt.referenc_img_file = f'data/mvtec_anomaly_detection/{opt.class_name}/train/good/000.png'
        # opt.referenc_img_file =  f'data/ref/{opt.class_name}/ref.png'
        # opt.referenc_img_file = f'natrual.JPEG'
        print(opt.class_name, opt.model_name, opt.k, opt.center_num, opt.sigma, opt.in_lay_num, opt.clu_lay_num)
        # print(opt.referenc_img_file)
        # opt.resume = r'result/RB_VIT_dir_res_ref_VGG/weight/capsule'
        opt.train_dataset = MVTecDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=True)
        opt.test_dataset = MVTecDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=False)
        opt.trainloader = DataLoader(opt.train_dataset, batch_size=1, shuffle=False)
        opt.testloader = DataLoader(opt.test_dataset, batch_size=1, shuffle=False)
        model = Model(opt)
        model.test()
    print(class_rocauc)
    value = list(class_rocauc.values())
    img_roc = [i[0] for i in value]
    pixel_roc = [i[1] for i in value]
    img_ap = [i[2] for i in value]
    pixel_ap = [i[3] for i in value]
    mean_img_roc = np.mean(np.array(img_roc))
    mean_pixel_roc = np.mean(np.array(pixel_roc))
    mean_img_ap = np.mean(np.array(img_ap))
    mean_pixel_ap = np.mean(np.array(pixel_ap))

    print(round(mean_img_roc, 3), round(mean_pixel_roc, 3), round(mean_img_ap, 3), round(mean_pixel_ap, 3))
