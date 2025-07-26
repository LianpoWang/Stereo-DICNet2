import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo)
from .attention import SelfAttnPropagation
from .geometry import flow_warp, compute_flow_with_depth_pose
from .reg_refine import BasicUpdateBlock_flow, BasicUpdateBlock_disp,BasicUpdateBlock
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

        #没有优化步骤
        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        # 有优化步骤
        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )
            self.refine_flow = BasicUpdateBlock_flow(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2,
                                           bilinear_up=task == 'depth',
                                           )
            self.refine_disp = BasicUpdateBlock_disp(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=1 ,
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



    def forward(self, img0, img1, img2):

        attn_type = None,
        attn_splits_list = (1),   #NNN分成2*2个窗口  划分的个数越少 能处理的位移越大    8
        corr_radius_list = (5),    #corr_radius_list = (-1), #-1代表全局匹配
        prop_radius_list = (-1),
        num_reg_refine = (8),  #1
        num_reg_refine = num_reg_refine[0]

        pred_bidir_flow = False,
        task = "flow",  #'stereo'
        pose = None,  # relative pose transform
        min_depth = 1. / 0.5,  # inverse depth range
        max_depth = 1. / 10,
        num_depth_candidates = 64,
        pred_bidir_depth = False,


        results_dict = {}
        flow_preds = []
        disp_preds = []

        # list of features, resolution low to high
        feature0_list, feature1_list, feature2_list = self.feature_extraction(img0, img1, img2)  # list of features

        flow = None
        disp = None


        if task != 'depth':
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales   #1
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1, feature2 = feature0_list[scale_idx], feature1_list[scale_idx],feature2_list[scale_idx]


            feature0_ori, feature1_ori, feature2_ori = feature0, feature1, feature2

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))   #8*(2**(1-1-0) = 8


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

            attn_splits = attn_splits_list[scale_idx]   # 2

            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx]   #-1
                prop_radius = prop_radius_list[scale_idx]



            # add position to features attn_splits = 2 feature_channels = 128
            feature0, feature1,feature2= feature_add_position(feature0, feature1, feature2, attn_splits, self.feature_channels)

           
            # Transformer
            featuret0, featuret1 = self.transformer(feature0, feature2,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

            features0, features2 = self.transformer(feature0, feature1,
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
                    #flow_pred = global_correlation_softmax(featuret0, featuret1, pred_bidir_flow)[0]
                    flow_pred = global_correlation_softmax(featuret0, featuret1)[0]
                    #disp_pred = global_correlation_softmax_stereo(features0, features2)[0]
                    disp_pred = global_correlation_softmax_stereo(features0, features2)[0] #迁移学习 变形图像作为参考图像 参考图像作为变形图像
                    

                else:  # local matching
                    flow_pred = local_correlation_softmax(featuret0, featuret1, corr_radius)[0]
                    #disp_pred = global_correlation_softmax_stereo(features0, features2)[0]
                    disp_pred = global_correlation_softmax_stereo(features0, features2)[0] #迁移学习 变形图像作为参考图像 参考图像作为变形图像
                    

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            disp = disp + disp_pred if disp is not None else disp_pred


            disp = disp.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only 训练执行 测试不执行
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

                disp_bilinear = self.upsample_flow(disp, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                disp_preds.append(disp_bilinear)


            # bilinear exclude the last one 不执行
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, featuret0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)
                disp_up = self.upsample_flow(disp, features0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                disp_preds.append(disp_up)
            #执行
            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution   直接输出预测的位移场
                    disp_pad = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # disp_pad:[B, 2, H, W] features0:[B, 128, H, W]
                    disp_up_pad = self.upsample_flow(disp_pad, features0)
                    disp_up = -disp_up_pad[:, :1]  # [B, 1, H, W]
                    disp_preds.append(disp_up)
                    flow_up = self.upsample_flow(flow, featuret0)
                    flow_preds.append(flow_up)


                else:
                    # #正常输出视差######--------结束
                    # disp_pad = torch.cat((-disp, torch.zeros_like(disp)),dim=1)  # disp_pad:[B, 2, H, W] features0:[B, 128, H, W]
                    # disp_up_pad = self.upsample_flow(disp_pad, features0)
                    # disp_up = -disp_up_pad[:, :1]  # [B, 1, H, W]
                    # disp_preds.append(disp_up)
                    # # 正常输出视差######--------结束
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        #print('flow:',flow.size())  #32*32
                        flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        #print('flow_up:', flow_up.size())  #256*256
                        flow_preds.append(flow_up)

                        disp_up = self.upsample_flow(disp, feature0, bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        disp_preds.append(disp_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):

                        flow = flow.detach()
                        disp = disp.detach()


                        if task == "stereo":
                            zeros = torch.zeros_like(disp)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                            proj = self.refine_proj(feature0)

                            net, inp = torch.chunk(proj, chunks=2, dim=1)

                            net = torch.tanh(net)
                            inp = torch.relu(inp)

                            net, up_mask, residual_disp = self.refine(net, inp, correlation, disp.clone(),
                                                                      )
                            disp = disp + residual_disp
                            disp = disp.clamp(min=0)  # positive


                        else:
                            # 下面的是优化光流的
                            correlation_flow = local_correlation_with_flow(
                                feature0_ori,
                                feature2_ori,
                                flow=flow,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                            proj = self.refine_proj(featuret0)

                            net, inp = torch.chunk(proj, chunks=2, dim=1)

                            net = torch.tanh(net)
                            inp = torch.relu(inp)

                            net, up_mask, residual_flow = self.refine_flow(net, inp, correlation_flow , flow.clone(),)
                            flow = flow + residual_flow

                            # 下面的是优化视差的
                            zeros = torch.zeros_like(disp)  # [B, 1, H, W]
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
                            correlation_disp = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  # [B, (2R+1)^2, H, W]
                            proj = self.refine_proj(feature0)

                            # correlation_disp = local_correlation_with_flow(
                            #     feature1_ori,
                            #     feature0_ori,
                            #     flow=displace,
                            #     local_radius=4,
                            # )  # [B, (2R+1)^2, H, W]
                            # proj = self.refine_proj(feature1)  #迁移学习 变形图像作为参考图像 参考图像作为变形图像
                            
                            net, inp = torch.chunk(proj, chunks=2, dim=1)
                            
                            net = torch.tanh(net)
                            inp = torch.relu(inp)
                            
                            net, up_mask, residual_disp = self.refine_disp(net, inp, correlation_disp, disp.clone(),
                                                                      )
                            disp = disp + residual_disp
                            disp = disp.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == "stereo":
                                disp_pad = torch.cat((-disp, torch.zeros_like(disp)),
                                                     dim=1)  # disp_pad:[B, 2, H, W] features0:[B, 128, H, W]

                                disp_up = upsample_flow_with_mask(disp_pad, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                disp_up = -disp_up[:, :1]

                                disp_preds.append(disp_up)


                            else:
                                # 下面的是优化光流的
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                flow_preds.append(flow_up)

                                # #只优化计算flow的时候 disp_preds列表的长度为0 这个是不做优化处理的视差
                                # disp_bilinear = self.upsample_flow(disp, None, bilinear=True,
                                #                                    upsample_factor=upsample_factor,
                                #                                    is_depth=task == 'depth')
                                # disp_preds.append(disp_bilinear)

                                # 下面的是优化视差的
                                disp_pad = torch.cat((-disp, torch.zeros_like(disp)),
                                                     dim=1)  # disp_pad:[B, 2, H, W] features0:[B, 128, H, W]
                                
                                disp_up = upsample_flow_with_mask(disp_pad, up_mask,
                                                                  upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                disp_up = -disp_up[:, :1]
                                
                                disp_preds.append(disp_up)
                                
                                # flow_bilinear = self.upsample_flow(flow, None, bilinear=True,
                                #                                    upsample_factor=upsample_factor,
                                #                                    is_depth=task == 'depth')
                                # flow_preds.append(flow_bilinear)


        #if task == 'stereo':

        for i in range(len(disp_preds)):
            disp_preds[i] = disp_preds[i].squeeze(1)  # [B, H, W]

        print("有局部优化的disp_preds长度：", len(disp_preds)) #5



        # results_dict.update({'flow_preds': flow_preds,'disp_preds': disp_preds})
        # return results_dict
        print('featuret0.type:',featuret0.shape)

        #return [flow_preds, disp_preds] #只优化计算flow的时候 disp_preds列表的长度为0
        return feature0_ori, feature1_ori
        #return [flow_preds] #只优化计算flow

        #return [disp_preds]  # 只计算disparity




