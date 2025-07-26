import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(feature0, feature1,
                               pred_bidir_flow=False,
                               ):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]
    feature0t = feature0.permute(0, 2, 1)
    feature1t = feature1.permute(0, 2, 1)
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]
    #correlation_NCC = correlation/torch.matmul(feature0, feature0t)/torch.matmul(feature1t, feature1).view(b, h, w, h, w)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]
    #correlation_NCC = correlation_NCC.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]
    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
    flow = correspondence - init_grid

    return flow, prob
    #correspondence_NCC = torch.matmul(correlation_NCC, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow


def local_correlation_softmax(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob


def local_correlation_with_flow(feature0, feature1,
                                flow,
                                local_radius,
                                padding_mode='zeros',
                                dilation=1,
                                ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation  # [B, H*W, (2R+1)^2, 2]

    # flow can be zero when using features after transformer
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(
            b, 2, -1).permute(0, 2, 1).unsqueeze(-2)  # [B, H*W, (2R+1)^2, 2]
    else:
        assert flow == 0.

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, (2R+1)^2, H, W]

    return corr


def global_correlation_softmax_stereo(feature0, feature1,
                                      ):
    # global correlation on horizontal direction
    b, c, h, w = feature0.shape

    x_grid = torch.linspace(0, w - 1, w, device=feature0.device)  # [W]

    feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
    feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

    correlation = torch.matmul(feature0, feature1) / (c ** 0.5)  # [B, H, W, W]

    # mask subsequent positions to make disparity positive
    mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)  # [W, W]
    valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]

    correlation[~valid_mask] = -1e9

    prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]

    correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

    # NOTE: unlike flow, disparity is typically positive
    disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]

    return disparity.unsqueeze(1), prob  # feature resolution


def local_correlation_softmax_stereo(feature0, feature1, local_radius,
                                     ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1).contiguous()  # [B, H*W, 2]

    local_h = 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(0, 0,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1), 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1), 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode='zeros', align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)]
    feature0_view = feature0.permute(0, 2, 3, 1).contiguous().view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)]

    correspondence = torch.matmul(prob.unsqueeze(-2),
                                  sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

    flow = correspondence - coords_init  # flow at feature resolution
    match_prob = prob

    flow_x = -flow[:, :1]  # [B, 1, H, W]

    return flow_x, match_prob




