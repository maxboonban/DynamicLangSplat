#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def compute_depth_loss(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))

def compute_flow_loss(
    render_flow_fwd, render_flow_bwd,
    fwd_flow, bwd_flow,
    fwd_flow_mask, bwd_flow_mask
    ):
    flow_loss = 0.

    fwd_flow = fwd_flow.permute(2, 0, 1)
    render_flow_fwd = render_flow_fwd[:2, ...]
    fwd_flow = fwd_flow / (torch.max(torch.sqrt(torch.square(fwd_flow).sum(-1))) + 1e-5)
    render_flow_fwd = render_flow_fwd / (torch.max(torch.sqrt(torch.square(render_flow_fwd).sum(-1))) + 1e-5)
    M = fwd_flow_mask.unsqueeze(0)
    fwd_flow_loss = torch.sum(torch.abs(fwd_flow - render_flow_fwd) * M) / (torch.sum(M) + 1e-8) / fwd_flow.shape[-1]
    flow_loss += fwd_flow_loss

    bwd_flow = bwd_flow.permute(2, 0, 1)
    render_flow_bwd = render_flow_bwd[:2, ...]
    bwd_flow = bwd_flow / (torch.max(torch.sqrt(torch.square(bwd_flow).sum(-1))) + 1e-5)
    render_flow_bwd = render_flow_bwd / (torch.max(torch.sqrt(torch.square(render_flow_bwd).sum(-1))) + 1e-5)
    M = bwd_flow_mask.unsqueeze(0)
    bwd_flow_loss = torch.sum(torch.abs(bwd_flow - render_flow_bwd) * M) / (torch.sum(M) + 1e-8) / bwd_flow.shape[-1]
    flow_loss += bwd_flow_loss
    return flow_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

