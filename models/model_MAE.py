import random

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from models.utils import get_2d_sincos_pos_embed
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F

class Adpative_MAE_k_center(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, clu_depth=1,
                 # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, center_num=8, sigma=2):
        super(Adpative_MAE_k_center, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.auxi_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.auxiliary = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(clu_depth)])
        self.norm_auxi = norm_layer(embed_dim)

        num_patches = self.patch_embed.num_patches

        self.center_num = center_num
        self.sigma = sigma
        self.cls_token = nn.Parameter(torch.zeros(1, center_num, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim, requires_grad=False))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        #
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding
        #
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        #
        # self.decoder_norm = norm_layer(decoder_embed_dim)

        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def generate_rand_noise(self, Batch, embed_dim, x):
        num_patches = self.patch_embed.num_patches
        noise_num = int(random.uniform(0, 0.75) * num_patches)
        self.noise_num = noise_num
        noise_index_list = [random.sample(range(num_patches), noise_num) for _ in range(Batch)]
        self.noise_index_list = noise_index_list
        tensor_defect = torch.zeros((Batch, num_patches, embed_dim))
        x_norm = x.norm(dim=2).unsqueeze(-1) / embed_dim
        tensor_defect[torch.arange(Batch).unsqueeze(-1), torch.tensor(noise_index_list, dtype=torch.long), :] = 1
        tensor_defect = tensor_defect.to(x.device)
        noise = torch.randn(Batch, num_patches, embed_dim).to(x.device) * x_norm * 5
        noise = tensor_defect * noise
        # noise = noise.to(device)
        return noise

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs



    def find_outliers(self, per_cluster_indices, per_anomaly_score, sigma, j):
        outlier_indices = []
        for k in range(self.center_num):
            anomaly_score_in_cluster, index_in_cluster = per_anomaly_score[per_cluster_indices == k], \
            torch.where(per_cluster_indices == k)[0]
            if len(anomaly_score_in_cluster) > 1:
                avg_distance = torch.mean(anomaly_score_in_cluster)
                std_distance = torch.std(anomaly_score_in_cluster)
                threshold = avg_distance + sigma * std_distance
                outliers = index_in_cluster[torch.where(anomaly_score_in_cluster >= threshold)[0]]
                outlier_indices.extend(outliers)
        return torch.tensor(outlier_indices, device=per_cluster_indices.device)

    def generate_mask(self, anomaly_score, cluster_index, sigma):
        B, N = anomaly_score.shape
        self.binary_anomaly = torch.ones([B, N], device=anomaly_score.device)
        mask_token = []
        for j, i in enumerate(range(B)):
            mask_token.append(self.find_outliers(cluster_index[i], anomaly_score[i], sigma, j))
        for i, index in enumerate(mask_token):
            if len(index) != 0:
                self.binary_anomaly[i, index] = 0.
        return self.binary_anomaly

    def rand_mask(self, anomaly_score):
        B, N = anomaly_score.shape
        self.binary_anomaly = torch.ones([B, N], device=anomaly_score.device)
        # self.binary_anomaly[anomaly_score >= threshold] = 0.
        noise = torch.rand(B, N, device=anomaly_score.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        mask_token = []
        for i in ids_shuffle:
            mask_len = int(N*random.uniform(0., 1.))
            # mask_len = int(N * 0.75)
            mask_token.append(i[:mask_len])
        for i, index in enumerate(mask_token):
            if len(index) != 0:
                self.binary_anomaly[i, index] = 0.

        return self.binary_anomaly


    def replace_mask(self, x, mask):
        masked_x = torch.where(mask.unsqueeze(-1) == 0., self.mask_token, x)
        return masked_x

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                    feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=torch.device("cuda:0"))
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature

        x, auxi_x_ori = self.patch_embed(x), self.auxi_embed(x)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(auxi_x_ori.shape[0], -1, -1)
        # x1 = x_detach + self.pos_embed
        if stage == 'train':
            auxi_x_noise = self.add_jitter(auxi_x_ori, 20, 1)
        else:
            auxi_x_noise = auxi_x_ori
        x1 = torch.cat((cls_tokens, auxi_x_noise+self.pos_embed), dim=1)
        for blk in self.auxiliary:
            x1 = blk(x1)
        self.auxiliary_feature = self.norm_auxi(x1[:, :self.center_num, :])

        # cos + l2
        if stage == 'train':
            self.diff_cos = torch.norm(self.auxiliary_feature.unsqueeze(2) - self.auxiliary_feature.unsqueeze(1), p=2,
                                       dim=-1) * (1. - F.cosine_similarity(self.auxiliary_feature.unsqueeze(2),
                                                                           self.auxiliary_feature.unsqueeze(1), dim=-1))

        cos_similarity = torch.norm((auxi_x_ori+self.pos_embed).unsqueeze(2) - self.auxiliary_feature.unsqueeze(1), p=2, dim=-1) * (
                    1. - F.cosine_similarity((auxi_x_ori+self.pos_embed).unsqueeze(2), self.auxiliary_feature.unsqueeze(1), dim=-1))

        # only L2
        # if stage == 'train':
        #     self.diff_cos = torch.norm(self.auxiliary_feature.unsqueeze(2) - self.auxiliary_feature.unsqueeze(1), p=2, dim=-1)
        #
        # cos_similarity = torch.norm((auxi_x_ori+self.pos_embed).unsqueeze(2) - self.auxiliary_feature.unsqueeze(1), p=2,dim=-1)
        # only cos
        # if stage == 'train':
        #     self.diff_cos = (1. - F.cosine_similarity(self.auxiliary_feature.unsqueeze(2),
        #                                                                    self.auxiliary_feature.unsqueeze(1), dim=-1))
        #
        # cos_similarity = (1. - F.cosine_similarity((auxi_x_ori+self.pos_embed).unsqueeze(2), self.auxiliary_feature.unsqueeze(1), dim=-1))


        self.anomaly_score, cluster_index = torch.min(cos_similarity, dim=2)
        if stage == 'train':
            mask = self.rand_mask(self.anomaly_score)
        else:
            mask = self.generate_mask(self.anomaly_score, cluster_index, self.sigma)
        masked_x = self.replace_mask(x, mask)
        if stage == 'train':
            masked_x = self.add_jitter(masked_x, 20, 1)
        masked_x = masked_x + self.pos_embed
        if stage=='train':
            for blk in self.blocks:
                # print(masked_x.shape)
                masked_x = blk(masked_x)
            masked_x = self.norm(masked_x)
            masked_x = self.inpainting_pred(masked_x)
            return masked_x, None
        elif stage=='test':
            for blk in self.blocks:
                masked_x = blk(masked_x)
            masked_x = self.norm(masked_x)
            masked_x = self.inpainting_pred(masked_x)
            return masked_x, None


    def forward_loss(self, imgs, pred, pred_nor):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        # pred = self.unpatchify(pred)
        N, L, _ = target.shape
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        dis_loss = (pred - target) ** 2
        dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)
        auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
                    self.center_num * self.center_num - self.center_num)
        loss_g = 5 * dir_loss.mean() + dis_loss.mean() + auxi_loss.mean()
        return loss_g

    def forward(self, imgs, stage):
        pred_mask, pred_normal = self.forward_encoder(imgs, stage)
        if stage == "train":
            loss = self.forward_loss(imgs, pred_mask, pred_normal)
        else:
            loss = 0.
        return loss, pred_mask, self.anomaly_score, self.binary_anomaly





