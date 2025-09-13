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

        # CNN Extract features
        self.backbone = CNNEncoder(output_dim=self.feature_channels, num_output_scales=self.num_scales)

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
            self.refine_flow = BasicUpdateBlock_flow(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2,
                                           bilinear_up=task == 'depth',
                                           )
            self.refine_disp = BasicUpdateBlock_disp(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 ,
                                           bilinear_up=task == 'depth',
                                           )

    def feature_extraction(self, img0, img1, img2):
        concat = torch.cat((img0, img1, img2), dim=0) 
        features = self.backbone(concat) 

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
        attn_splits_list = (4),  
        corr_radius_list = (5),   
        prop_radius_list = (-1),
        num_reg_refine = (12),  
        num_reg_refine = num_reg_refine[0]

        pred_bidir_flow = False,
        task = "flow", 
        pose = None,
        min_depth = 1. / 0.5, 
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
            assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales   
        else:
            assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales == 1

        for scale_idx in range(self.num_scales):
            feature0, feature1, feature2 = feature0_list[scale_idx], feature1_list[scale_idx],feature2_list[scale_idx]


            feature0_ori, feature1_ori, feature2_ori = feature0, feature1, feature2

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))  


            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
                disp = F.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=True) * 2


            if disp is not None:
              disp = disp.detach()
              zeros = torch.zeros_like(disp)  
              displace = torch.cat((-disp, zeros), dim=1) 
              feature1 = flow_warp(feature1, displace) 



            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  

            attn_splits = attn_splits_list[scale_idx]  

            if task != 'depth':
                corr_radius = corr_radius_list[scale_idx] 
                prop_radius = prop_radius_list[scale_idx]

          
            N,C,H,W = img0.shape
            coords0 = torch.meshgrid(torch.arange(H//8,device=img0.device),torch.arange(W//8,device=img0.device))
            coords0 = torch.stack(coords0[::-1],dim=0).float()
            coords0 = coords0[None].repeat(N,1,1,1)

            coords1 = torch.meshgrid(torch.arange(H//8,device=img0.device),torch.arange(W//8,device=img0.device))
            coords1 = torch.stack(coords1[::-1],dim=0).float()
            coords1 = coords1[None].repeat(N,1,1,1)

            coords2 = torch.meshgrid(torch.arange(W//8,device=img0.device))
            coords2 = torch.stack(coords2[::-1],dim=0).float()
            coords2 = coords2[None].repeat(N,1,1,1)

            flow = coords1 - coords0
            disp = coords2 - coords0



            disp = disp.clamp(min=0)  # positive disparity

           
            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    disp_pad = torch.cat((-disp, torch.zeros_like(disp)), dim=1) 
                    disp_up_pad = self.upsample_flow(disp_pad, feature0)
                    disp_up = -disp_up_pad[:, :1] 
                    disp_preds.append(disp_up)
                    flow_up = self.upsample_flow(flow, feature0)
                    flow_preds.append(flow_up)


                else:
                  
                    if self.training:
                        #print('flow:',flow.size()) 
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
                            zeros = torch.zeros_like(disp)  
                            # NOTE: reverse disp, disparity is positive
                            displace = torch.cat((-disp, zeros), dim=1) 
                            correlation = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  
                            proj = self.refine_proj(feature0)
                            net, inp = torch.chunk(proj, chunks=2, dim=1)
                            net = torch.tanh(net)
                            inp = torch.relu(inp)
                            net, up_mask, residual_disp = self.refine(net, inp, correlation, disp.clone(),)                                                          
                            disp = disp + residual_disp
                            disp = disp.clamp(min=0)  # positive


                        else:
                            correlation_flow = local_correlation_with_flow(
                                feature0_ori,
                                feature2_ori,
                                flow=flow,
                                local_radius=4,
                            )  
                            #proj = self.refine_proj(featuret0)
                            #print('flow:',flow.size())  
                            proj = self.refine_proj(feature0)


                            net, inp = torch.chunk(proj, chunks=2, dim=1)
                            net = torch.tanh(net)
                            inp = torch.relu(inp)
                            net, up_mask, residual_flow = self.refine_flow(net, inp, correlation_flow , flow.clone(),)
                            flow = flow + residual_flow


                            zeros = torch.zeros_like(disp)  
                            # NOTE: reverse disp, disparity is positive
                            #displace = torch.cat((-disp, zeros), dim=1) 
                            displace = disp  
                    
                            correlation_disp = local_correlation_with_flow(
                                feature0_ori,
                                feature1_ori,
                                flow=displace,
                                local_radius=4,
                            )  
                            proj = self.refine_proj(feature0)                                          
                            net, inp = torch.chunk(proj, chunks=2, dim=1)
                            
                            net = torch.tanh(net)
                            inp = torch.relu(inp)
                            
                            net, up_mask, residual_disp = self.refine_disp(net, inp, correlation_disp, disp.clone(),)
                                                           
                            disp = disp + residual_disp
                            disp = disp.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            if task == "stereo":
                                disp_pad = torch.cat((-disp, torch.zeros_like(disp)),
                                                     dim=1) 

                                disp_up = upsample_flow_with_mask(disp_pad, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                disp_up = -disp_up[:, :1]

                                disp_preds.append(disp_up)


                            else:
                                flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                flow_preds.append(flow_up)

                                disp_pad = torch.cat((-disp, torch.zeros_like(disp)),
                                                     dim=1) 
                                
                                disp_up = upsample_flow_with_mask(disp_pad, up_mask,
                                                                  upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')
                                disp_up = -disp_up[:, :1]
                                
                                disp_preds.append(disp_up)
                           

        return [flow_preds, disp_preds] 
        
