import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock
from .utils import normalize_img, feature_add_position, upsample_flow_with_mask



class UniDIC(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 ):

        super(UniDIC, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN 特征提取
        self.backbone = CNNEncoder(output_dim=self.feature_channels, num_output_scales=self.num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )

    def feature_extraction(self, img0, img1, img2):
        concat = torch.cat((img0, img1, img2), dim=0)  # [3B, C, H, W]
        features = self.backbone(concat)  # list of [3B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1, feature2 = [], [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 3, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])
            feature2.append(chunks[2])

        return feature0, feature1, feature2

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):

        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    print('前向传播')
    def forward(self, img0, img1, img2):
        print('前向传播')
        attn_type = None,
        attn_splits_list = [2],
        corr_radius_list = [-1],
        prop_radius_list = [-1],
        num_reg_refine = 1,
        pred_bidir_flow = False,
        task = 'flow',
        pose = None,  # relative pose transform
        min_depth = 1. / 0.5,  # inverse depth range
        max_depth = 1. / 10,
        num_depth_candidates = 64,
        pred_bidir_depth = False,

        print('1111')
        results_dict = {}
        flow_preds = []
        disp_preds = []

        # list of features, resolution low to high
        feature0_list, feature1_list, feature2_list = self.feature_extraction(img0, img1, img2)  # list of features

        flow = None
        disp = None


        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1, feature2 = feature0_list[scale_idx], feature1_list[scale_idx],feature2_list[scale_idx]


            feature0_ori, feature1_ori, feature2_ori = feature0, feature1, feature2

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))   #8*(2**(1-1-0)=8


            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
                disp = F.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=True) * 2


            if disp is not None:
              disp = disp.detach()
              zeros = torch.zeros_like(disp)  # [B, 1, H, W]
              displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
              feature1 = flow_warp(feature1, displace)  # [B, C, H, W]



            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]
                prop_radius = prop_radius_list[scale_idx]



            # add position to features
            feature0, feature1,feature2= feature_add_position(feature0, feature1, feature2, attn_splits, self.feature_channels)

            # Transformer
            featuret0, featuret1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            features0, features2 = self.transformer(feature0, feature2,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            # correlation and softmax
            if task == 'depth':
                # first generate depth candidates
                b, _, h, w = feature0.size()
                depth_candidates = torch.linspace(min_depth, max_depth, num_depth_candidates).type_as(feature0)
                depth_candidates = depth_candidates.view(1, num_depth_candidates, 1, 1).repeat(b, 1, h,
                                                                                               w)  # [B, D, H, W]


            else:
                if corr_radius == -1:  # global matching
                    flow_pred = global_correlation_softmax(featuret0, featuret1, pred_bidir_flow)[0]
                    disp_pred = global_correlation_softmax_stereo(features0, features2)[0]

                else:  # local matching
                    flow_pred = local_correlation_softmax(featuret0, featuret1, corr_radius)[0]
                    disp_pred = local_correlation_softmax_stereo(features0, features2, corr_radius)[0]


            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            disp = disp + disp_pred if disp is not None else disp_pred


            disp = disp.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

                disp_bilinear = self.upsample_flow(disp, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                disp_preds.append(disp_bilinear)


            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, featuret0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)
                disp_up = self.upsample_flow(disp, features0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                disp_preds.append(disp_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    disp_pad = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
                    disp_up_pad = self.upsample_flow(disp_pad, features0)
                    disp_up = -disp_up_pad[:, :1]  # [B, 1, H, W]
                    disp_preds.append(disp_up)

                    flow_up = self.upsample_flow(flow, featuret0)
                    flow_preds.append(flow_up)


                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()

                        if task == 'stereo':
                            zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                        elif task == 'depth':
                            if pred_bidir_depth and refine_iter_idx == 0:
                                intrinsics_curr = intrinsics_curr.repeat(2, 1, 1)
                                pose = torch.cat((pose, torch.inverse(pose)), dim=0)

                                feature0_ori, feature1_ori = torch.cat((feature0_ori, feature1_ori),
                                                                       dim=0), torch.cat((feature1_ori,
                                                                                          feature0_ori), dim=0)

                            flow_from_depth = compute_flow_with_depth_pose(1. / flow.squeeze(1),
                                                                           intrinsics_curr,
                                                                           extrinsics_rel=pose,
                                                                           )

                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow_from_depth,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        else:
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]

                        proj = self.refine_proj(feature0)

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone(),
                                                                  )

                        if task == 'depth':
                            flow = (flow - residual_flow).clamp(min=min_depth, max=max_depth)
                        else:
                            flow = flow + residual_flow

                        if task == 'stereo':
                            flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == 'depth':
                                if refine_iter_idx < num_reg_refine - 1:
                                    # bilinear upsampling
                                    flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                                 upsample_factor=upsample_factor,
                                                                 is_depth=True)
                                else:
                                    # last one convex upsampling
                                    # NOTE: clamp depth due to the zero padding in the unfold in the convex upsampling
                                    # pad depth to 2 channels as flow
                                    depth_pad = torch.cat((flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                                    depth_up_pad = self.upsample_flow(depth_pad, feature0,
                                                                      is_depth=True).clamp(min=min_depth,
                                                                                           max=max_depth)
                                    flow_up = depth_up_pad[:, :1]  # [B, 1, H, W]

                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

        #if task == 'stereo':
        print(len(disp_preds))
        for i in range(len(disp_preds)):
            disp_preds[i] = disp_preds[i].squeeze(1)  # [B, H, W]



        #results_dict.update({'flow_preds': flow_preds,'disp_preds': disp_preds})
        #return results_dict
        print('1111')
        return [flow_preds, disp_preds]

    print('前向传播')

