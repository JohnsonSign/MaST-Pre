import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from SHOT import *
from transformer import *
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2


class ContrastiveLearningModel(nn.Module):
    def __init__(self, radius=0.1, nsamples=32, spatial_stride=32,                            # P4DConv: spatial
                 temporal_kernel_size=3, temporal_stride=3,                                   # P4DConv: temporal
                 en_emb_dim=1024, en_depth=10, en_heads=8, en_head_dim=256, en_mlp_dim=2048,  # encoder
                 de_emb_dim=512,  de_depth=4,  de_heads=8, de_head_dim=256, de_mlp_dim=1024,  # decoder 
                 mask_ratio=0.6,
                 num_classes=60,
                 dropout1=0.05,
                 dropout_cls=0.5,
                 pretraining=True,
                 vis=False,
                 ):
        super(ContrastiveLearningModel, self).__init__()

        self.pretraining = pretraining

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[en_emb_dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        
        # encoder        
        self.encoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=en_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.encoder_transformer = Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
        self.encoder_norm = nn.LayerNorm(en_emb_dim)

        self.vis = vis
        self.nsamples = nsamples
        self.tk = temporal_kernel_size
        self.shotL = getDescriptorLength(elevation_divisions=2,
                                        azimuth_divisions=4
        )
        if self.pretraining:
            
            # Mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.mask_token, std=.02)

            # decoder
            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=de_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

            self.decoder_transformer = Transformer(de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1)
            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            # points_predictor
            self.points_predictor = nn.Conv1d(de_emb_dim, 3 * nsamples * temporal_kernel_size, 1)
            self.shot_predictor = nn.Conv1d(de_emb_dim, self.shotL * (temporal_kernel_size-1), 1)

            # loss
            self.criterion_dist = ChamferDistanceL2().cuda()
            self.criterion_shot = torch.nn.SmoothL1Loss().cuda()

            self.mask_ratio = mask_ratio

        else:
            # PointMAE mlp_head
            self.cls_token = nn.Parameter(torch.zeros(1, 1, en_emb_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, en_emb_dim))

            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

            self.mlp_head = nn.Sequential(
                nn.Linear(en_emb_dim*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_cls),
                nn.Linear(256, num_classes)
            )

        # self.apply(self._init_weights)


    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x):
        B, G, _ = x.shape

        if self.mask_ratio == 0:
            return torch.zeros(x.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(x.device) # B G


    def forward_encoder(self, x):
        # [B, L, N, 3]
        xyzs, features, xyzs_neighbors, shot_descriptors = self.tube_embedding(x)  
        # [B, L, N, 3] [B, L, C, N] [B, L, N, tk, nn, 3] [B, L, N, tk, shotL]
        
        features = features.permute(0, 1, 3, 2)                                              # [B, L, N, C]        
        batch_size, L, N, C = features.shape

        # xyzt position
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]                # L*[B, N, 3]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((batch_size, N, 1), dtype=torch.float32, device=x.device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)                                            # [B, L, N, 4]

        # Token sequence
        xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 4]
        features = torch.reshape(input=features, shape=(batch_size, L*N, C))

        if self.pretraining:
            # Targets
            xyzs_neighbors = torch.reshape(input=xyzs_neighbors, shape=(batch_size, L*N, self.tk, self.nsamples, 3)) # [B, L*N, tk, nn, 3]
            shot_descriptors = torch.reshape(input=shot_descriptors, shape=(batch_size, L*N, self.tk, self.shotL))   # [B, L*N, tk, shotL]

            # Masking
            bool_masked_pos = self.random_masking(xyzts)       # [B, L*N]   Vis=0 Mask=1
            
            # Encoding the visible part
            fea_emb_vis = features[~bool_masked_pos].reshape(batch_size, -1, C)
            pos_emb_vis = xyzts[~bool_masked_pos].reshape(batch_size, -1, 4)
            
            pos_emb_vis = self.encoder_pos_embed(pos_emb_vis.permute(0, 2, 1)).permute(0, 2, 1)

            fea_emb_vis = fea_emb_vis + pos_emb_vis

            fea_emb_vis = self.encoder_transformer(fea_emb_vis)
            fea_emb_vis = self.encoder_norm(fea_emb_vis)

            return fea_emb_vis, bool_masked_pos, xyzts, xyzs_neighbors, shot_descriptors

        else:
            xyzts = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cls_pos = self.cls_pos.expand(batch_size, -1, -1)

            features = torch.cat((cls_tokens, features), dim=1)
            xyzts = torch.cat((cls_pos, xyzts), dim=1)

            embedding = xyzts + features

            output = self.encoder_transformer(embedding)
            output = self.encoder_norm(output)

            concat_f = torch.cat([output[:, 0], output[:, 1:].max(1)[0]], dim=-1)

            output = self.mlp_head(concat_f)

            return output       


    def forward_decoder(self, emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors):
        emb_vis = self.decoder_embed(emb_vis)
        batch_size, N_vis, C_decoder = emb_vis.shape

        pos_emd_vis = xyzts[~mask].reshape(batch_size, -1, 4)
        pos_emd_mask = xyzts[mask].reshape(batch_size, -1, 4)

        pos_emd_vis = self.decoder_pos_embed(pos_emd_vis.permute(0, 2, 1)).permute(0, 2, 1)
        pos_emd_mask = self.decoder_pos_embed(pos_emd_mask.permute(0, 2, 1)).permute(0, 2, 1)

        _,N_masked,_ = pos_emd_mask.shape

        # append masked tokens to sequence
        mask_tokens = self.mask_token.expand(batch_size, N_masked, -1)
        emb_all = torch.cat([emb_vis, mask_tokens], dim=1)
        pos_all = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        emb_all = emb_all + pos_all

        emb_all = self.decoder_transformer(emb_all)  # [B, L*N, C]
        emb_all = self.decoder_norm(emb_all)

        masked_emb = emb_all[:, -N_masked:, :]       # [B, M, C]
        masked_emb = masked_emb.transpose(1, 2)      # [B, C, M]

        # reconstruct points
        pre_points = self.points_predictor(masked_emb).transpose(1, 2)   

        pre_points = pre_points.reshape(batch_size*N_masked, self.tk, self.nsamples, 3)                     # [B*M, tk, nn, 3]
        pred_list = torch.split(tensor=pre_points, split_size_or_sections=1, dim=1)     
        pred_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in pred_list]                     # tk*[B*M, nn, 3]

        # forward Loss
        gt_points = xyzs_neighbors[mask].reshape(batch_size*N_masked, self.tk, self.nsamples, 3)            # [B*M, tk, nn, 3]
        gt_points_list = torch.split(tensor=gt_points, split_size_or_sections=1, dim=1)
        gt_points_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in gt_points_list]           # tk*[B*M, nn, 3]

        point_loss = 0
        for tk_i in range(self.tk):
            point_loss += self.criterion_dist(pred_list[tk_i], gt_points_list[tk_i])
        point_loss = point_loss / self.tk

        # reconstruct shot
        pre_shot = self.shot_predictor(masked_emb).transpose(1, 2)   
        pre_shot = pre_shot.reshape(batch_size*N_masked, (self.tk-1), self.shotL)                           # [B*M, tk-1, shotL]
        pre_shot_list = torch.split(tensor=pre_shot, split_size_or_sections=1, dim=1)     
        pre_shot_list = [torch.squeeze(input=shot, dim=1).contiguous() for shot in pre_shot_list]           # (tk-1)*[B*M, shotL]

        gt_shot = shot_descriptors[mask].reshape(batch_size*N_masked, self.tk, self.shotL)                  # [B*M, tk, shotL]
        gt_shot_list = torch.split(tensor=gt_shot, split_size_or_sections=1, dim=1)     
        gt_shot_list = [torch.squeeze(input=shot, dim=1).contiguous() for shot in gt_shot_list]             # tk*[B*M, shotL]

        shot_loss = 0
        for tk_i in range(self.tk-1):
            shot_loss += self.criterion_shot(pre_shot_list[tk_i], gt_shot_list[tk_i+1]-gt_shot_list[tk_i])
        shot_loss = shot_loss / (self.tk-1)

        loss = point_loss + shot_loss

        if self.vis:
            vis_points = xyzs_neighbors[~mask].reshape(batch_size, -1, self.tk, self.nsamples, 3) # [B, L*N-m, tk*nn, 3]
            pre_points = pre_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)           
            gt_points = gt_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)
            return pre_points, gt_points, vis_points, mask
        else:
            return loss


    def forward(self, clips):
        # [B, L, N, 3]
        if self.pretraining:
            emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors = self.forward_encoder(clips)

            if self.vis:
                pre_points, gt_points, vis_points, mask = self.forward_decoder(emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors)
                return pre_points, gt_points, vis_points, mask
            else:
                loss = self.forward_decoder(emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors)
                return loss
        else:
            output = self.forward_encoder(clips)
            return output


