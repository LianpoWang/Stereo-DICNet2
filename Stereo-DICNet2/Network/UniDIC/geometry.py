import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float() 

    grid = grid[None].repeat(b, 1, 1, 1) 

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float() 

    return grid


def normalize_coords(coords, h, w):
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    if sample_coords.size(1) != 2: 
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1) 

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow) 
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow) 

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1) 
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def back_project(depth, intrinsics):
    # Back project 2D pixel coords to 3D points

    b, h, w = depth.shape
    grid = coords_grid(b, h, w, homogeneous=True, device=depth.device) 

    intrinsics_inv = torch.inverse(intrinsics)  

    points = intrinsics_inv.bmm(grid.view(b, 3, -1)).view(b, 3, h, w) * depth.unsqueeze(1)  

    return points


def camera_transform(points_ref, extrinsics_ref=None, extrinsics_tgt=None, extrinsics_rel=None):

    b, _, h, w = points_ref.shape

    if extrinsics_rel is None:
        extrinsics_rel = torch.bmm(extrinsics_tgt, torch.inverse(extrinsics_ref)) 

    points_tgt = torch.bmm(extrinsics_rel[:, :3, :3],
                           points_ref.view(b, 3, -1)) + extrinsics_rel[:, :3, -1:] 

    points_tgt = points_tgt.view(b, 3, h, w) 

    return points_tgt


def reproject(points_tgt, intrinsics, return_mask=False):
    # reproject to target view
    # points_tgt: [B, 3, H, W]
    # intrinsics: [B, 3, 3]

    b, _, h, w = points_tgt.shape

    proj_points = torch.bmm(intrinsics, points_tgt.view(b, 3, -1)).view(b, 3, h, w)

    X = proj_points[:, 0]
    Y = proj_points[:, 1]
    Z = proj_points[:, 2].clamp(min=1e-3)

    pixel_coords = torch.stack([X / Z, Y / Z], dim=1).view(b, 2, h, w) 

    if return_mask:
        # valid mask in pixel space
        mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] <= (w - 1)) & (
                pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] <= (h - 1)) 

        return pixel_coords, mask

    return pixel_coords


def reproject_coords(depth_ref, intrinsics, extrinsics_ref=None, extrinsics_tgt=None, extrinsics_rel=None,
                     return_mask=False):
    # Compute reprojection sample coords
    points_ref = back_project(depth_ref, intrinsics)  
    points_tgt = camera_transform(points_ref, extrinsics_ref, extrinsics_tgt, extrinsics_rel=extrinsics_rel)

    if return_mask:
        reproj_coords, mask = reproject(points_tgt, intrinsics,
                                        return_mask=return_mask) 

        return reproj_coords, mask

    reproj_coords = reproject(points_tgt, intrinsics,
                              return_mask=return_mask) 

    return reproj_coords


def compute_flow_with_depth_pose(depth_ref, intrinsics,
                                 extrinsics_ref=None, extrinsics_tgt=None, extrinsics_rel=None,
                                 return_mask=False):
    b, h, w = depth_ref.shape
    coords_init = coords_grid(b, h, w, device=depth_ref.device) 

    if return_mask:
        reproj_coords, mask = reproject_coords(depth_ref, intrinsics, extrinsics_ref, extrinsics_tgt,
                                               extrinsics_rel=extrinsics_rel,
                                               return_mask=return_mask)  
        rigid_flow = reproj_coords - coords_init

        return rigid_flow, mask

    reproj_coords = reproject_coords(depth_ref, intrinsics, extrinsics_ref, extrinsics_tgt,
                                     extrinsics_rel=extrinsics_rel,
                                     return_mask=return_mask) 

    rigid_flow = reproj_coords - coords_init

    return rigid_flow
