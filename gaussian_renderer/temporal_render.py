# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from scene.temporal_gaussian_model import TemporalGaussianModel, batch_SH_rotate
from utils.sh_utils import eval_sh
from typing import Optional

NUM_CHANNELS = 6

# pts: Nx3
# pts_left: Nx3
# pts_right: Nx3
def compute_sf_loss(
    pts: torch.Tensor,  
    pts_left: torch.Tensor,
    pts_right: torch.Tensor,
    pts_canon: torch.Tensor,
    mask=None
): 
    if mask is not None:
        #mask = mask[:, 0].bool()
        pts = pts[mask]
        pts_left = pts_left[mask]
        pts_right = pts_right[mask]
        pts_canon = pts_canon[mask]
    scene_flow_left = pts_left - pts # from left to current
    scene_flow_right = pts - pts_right # from current to right
    sm_loss = torch.mean(torch.abs(scene_flow_right - scene_flow_left))
    #return torch.mean(torch.abs(scene_flow_world[:-1, :] - scene_flow_world[1:, :]))
    st_loss = torch.mean(
        torch.abs(torch.cat([
            pts_canon - pts,
            pts_canon - pts_left,
            pts_canon - pts_right
            ], dim=0)))
    return sm_loss, st_loss

# pts: Nx3
# pts_left: Nx3
# pts_right: Nx3
def compute_scale_loss(
    scales: torch.Tensor,  
    scales_left: torch.Tensor,
    scales_right: torch.Tensor,
    scales_canon: torch.Tensor,
    mask=None
): 
    if mask is not None:
        #mask = mask[:, 0].bool()
        scales = scales[mask]
        scales_left = scales_left[mask]
        scales_right = scales_right[mask]
        scales_canon = scales_canon[mask]
    #scene_flow_left = pts_left - pts # from left to current
    #scene_flow_right = pts - pts_right # from current to right
    #sm_loss = torch.mean(torch.abs(scene_flow_right - scene_flow_left))
    sm_loss = torch.mean(torch.abs(scales**2 - scales_left * scales_right))
    #return torch.mean(torch.abs(scene_flow_world[:-1, :] - scene_flow_world[1:, :]))
    st_loss = torch.mean(
        torch.abs(torch.cat([
            scales_canon-scales,
            scales_canon-scales_left,
            scales_canon-scales_right
           ], dim=0))
    )
    #st_loss = 
    return sm_loss, st_loss

def temporal_render(viewpoint_camera, pc : TemporalGaussianModel, pipe, bg_color : torch.Tensor, disable_offscale, disable_offopa, disable_morph, multiply_offopa, disable_deform, visualize = False, scaling_modifier = 1.0, override_color = None,
    return_flow=False, return_sep=False, return_depth=False, anneal_band_iteration=None,
    sf_reg: bool = False,
    #motion_small: bool = False,
    motion_gap: float = 1e-3,
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #assert False, [disable_offopa, multiply_offopa]
    
    time = viewpoint_camera.time # float between 0 and 1
    if pc.use_ResFields:
        frame_id = viewpoint_camera.frame_id
    else:
        frame_id = None

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    '''
    assert False, "this part gradient is wrong for now"
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    '''
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    motion_tensor = None
    cov3D_precomp = None
    colors_precomp = None
    shs = pc.get_features
    dino = pc.get_features_dino
    # clip = pc.get_features_clip
    #else:
    sm_loss, st_loss = 0., 0.
    if disable_deform:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        if sf_reg or return_flow or visualize:
            means3D_left, opacity_left, scales_left, rotations_left = \
                pc.get_xyz, pc.get_opacity, pc.get_scaling, pc.get_rotation \
                #pc.get_deformed(time-motion_gap, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
            means3D_right, opacity_right, scales_right, rotations_right = \
                pc.get_xyz, pc.get_opacity, pc.get_scaling, pc.get_rotation 
            mask = pc._isstatic[:, 0].bool() 
            if not pc.enable_static:
                mask = (mask.float() * 0. + 1.).bool() 
            #mask = torch.zeros_like(pc._isstatic[:, 0]).bool().to(pc._isstatic.device) 
            #if not pc.enable_static:
            #    mask = (mask.float() * 0. + 1.).bool() 
                
            scene_flow_left = means3D_left - means3D # from left to current
            scene_flow_right = means3D - means3D_right # from current to right
            #motion_tensor = -torch.ones_like(means3D).to(means3D.device)
    else:
        # get means3D: the deformed xyz based on time
        # get opacities: the deformed opacity based on time
        # get scales: the deformed scales based on time
        # get rotations: the deformed rotation based on time
        if pc.separate_offsh:
            assert False, "motion regularizers not supported yet"
            means3D, opacity, scales, rotations, dfeat_dc, dfeat_extra = pc.get_deformed(time, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
            #features_dc = self._features_dc
            #features_rest = self._features_rest
            shs = torch.cat((pc._features_dc+dfeat_dc, 
            pc._features_rest+dfeat_extra), dim=1)
            
        
        elif pc.enable_offsh:
            #assert False, "motion regularizers not supported yet"
            means3D, opacity, scales, rotations, dfeat, dpos, dscale, drot = pc.get_deformed(time, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
            shs += dfeat
            if sf_reg or return_flow or visualize or pc.dynamic_sep:
                means3D_left, opacity_left, scales_left, rotations_left, dfeat_left, dpos_left, dscale_left, drot_left = pc.get_deformed(time-motion_gap, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                means3D_right, opacity_right, scales_right, rotations_right, dfeat_left, dpos_right, dscale_right, drot_right =  pc.get_deformed(time+motion_gap, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                mask = pc._isstatic[:, 0].bool() 
                if not pc.enable_static:
                    mask = (mask.float() * 0. + 1.).bool() 
                
                scene_flow_left = means3D_left - means3D # from left to current
                scene_flow_right = means3D - means3D_right # from current to right
                motion_tensor = torch.maximum(torch.abs(scene_flow_left), torch.abs(scene_flow_right)) * pc._isstatic
                
            
            if sf_reg:    
                sm_loss = torch.mean(torch.abs(scene_flow_right - scene_flow_left)[mask])
                st_loss = torch.mean(
                        #torch.norm(dpos, dim=1) + 
                        torch.abs(
                            torch.cat([
                                (dpos_left-dpos)[mask], 
                                (dpos_right-dpos)[mask]
                                ], dim=0),
                                ) 
                        #+
                        #torch.norm(dscale, dim=1) +
                        #torch.norm(dscale_left, dim=1) +
                        #torch.norm(dscale_right, dim=1) 
                        #+
                        #torch.norm(drot, dim=1) +
                        #torch.norm(drot_left, dim=1) +
                        #torch.norm(drot_right, dim=1)
                    ) 

            #assert False, [shs.shape, dfeat.shape]
        else:
            means3D, opacity, scales, rotations, dpos, dscale, drot = pc.get_deformed(time, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
            if sf_reg or return_flow or visualize or pc.dynamic_sep:
                means3D_left, opacity_left, scales_left, rotations_left, dpos_left, dscale_left, drot_left = pc.get_deformed(time-motion_gap, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                means3D_right, opacity_right, scales_right, rotations_right, dpos_right, dscale_right, drot_right =  pc.get_deformed(time+motion_gap, frame_id, disable_offscale, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                mask = pc._isstatic[:, 0].bool() 
                if not pc.enable_static:
                    mask = (mask.float() * 0. + 1.).bool() 
                
                scene_flow_left = means3D_left - means3D # from left to current
                scene_flow_right = means3D - means3D_right # from current to right

                motion_tensor = torch.maximum(torch.abs(scene_flow_left), torch.abs(scene_flow_right))* pc._isstatic
                #assert False, torch.all(motion_tensor == 0.)
            if sf_reg:    
                sm_loss = torch.mean(torch.abs(scene_flow_right - scene_flow_left)[mask])
                st_loss = torch.mean(
                        #torch.norm(dpos, dim=1) + 
                        torch.abs(
                            torch.cat([
                                (dpos_left-dpos)[mask], 
                                (dpos_right-dpos)[mask]
                                ], dim=0),
                                ) 
                        #+
                        #torch.norm(dscale, dim=1) +
                        #torch.norm(dscale_left, dim=1) +
                        #torch.norm(dscale_right, dim=1) 
                        #+
                        #torch.norm(drot, dim=1) +
                        #torch.norm(drot_left, dim=1) +
                        #torch.norm(drot_right, dim=1)
                    ) 

                #sm_loss, st_loss = compute_sf_loss(
                #    pts=means3D, pts_left=means3D_left,
                #    pts_right=means3D_right, pts_canon=pc.get_xyz,
                #    mask=mask)
                #sm_loss_scale, st_loss_scale = compute_scale_loss(
                #    scales=scales, scales_left=scales_left,
                #    scales_right=scales_right,
                #    mask=pc._isstatic
                #) # Too slow!
                #sm_loss += sm_loss_scale
                #scales_canon = pc.get_scaling
                #st_loss += torch.mean(
                #    torch.abs(
                #        torch.cat([
                #            (scales_canon - scales)[mask],
                #            (scales_canon - scales_left)[mask],
                #            (scales_canon - scales_right)[mask]
                #        ], dim=0))
                #)      
            #if True:
            #    scales = pc.get_scaling

        if pc.rotate_sh:
            shs = batch_SH_rotate(
                q=drot,
                features=shs)


    
    # get mean2D: screenspace gradient storage
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points


    #assert False, "Add static part here!"
    scaling_tensor = None
    opacity_tensor = None
    rendered_depth = None
    if pc.enable_static:
        static_filter = pc._isstatic
        #assert False, [shs.shape, static_filter.shape, pc.get_features.shape]
        
        sh_concat = shs * static_filter[..., None] + pc.get_features * (1.-static_filter[..., None])
        shs_view = sh_concat.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        xyz_concat = means3D * static_filter + pc.get_xyz * (1.-static_filter)
        dir_pp = (xyz_concat - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        dino_features = dino * static_filter[..., None] + pc.get_features_dino * (1.-static_filter[..., None])
        dino_features = dino_features.squeeze()
        # clip_features = clip * static_filter[..., None] + pc.get_features_clip * (1.-static_filter[..., None])
        # clip_features = clip_features.squeeze()
        
        features_precomp = torch.cat((colors_precomp, dino_features), dim=1)
        # features_precomp = torch.cat((colors_precomp, dino_features, clip_features), dim=1)
        
        rendered_features, radii, rendered_depth = rasterizer(
            means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
            means2D = means2D,
            shs = None,
            colors_precomp = features_precomp,
            opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
            scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
            rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
            cov3D_precomp = cov3D_precomp)
        
        rendered_image = rendered_features[:3, :, :]
        rendered_dino = rendered_features[3:6, :, :]
        # rendered_clip = rendered_features[6:-1, :, :]
        
        #print(torch.sum(static_filter), means3D.shape[0])
        #means3D[static_filter] = pc.get_xyz[static_filter]
        #opacity[static_filter] = pc.get_opacity[static_filter]
        #scales[static_filter] = pc.get_scaling[static_filter]
        #rotations[static_filter] = pc.get_rotation[static_filter]
        #shs[static_filter] = pc.get_features[static_filter]
        if pc.ewa_prune:
            with torch.no_grad():
                scaling_tensor = (scales * static_filter + pc.get_scaling * (1.-static_filter))
                scaling_tensor = torch.norm(scaling_tensor, dim=-1, keepdim=True)
                opacity_tensor = (opacity * static_filter + pc.get_opacity * (1.-static_filter))

        rendered_sep = None
        if return_sep:
            try:
                rendered_sep, _, _ = rasterizer(
                    means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
                    means2D = means2D,
                    shs = None,
                    colors_precomp = static_filter.view(-1, 1).repeat(1, NUM_CHANNELS),
                    opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
                    scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
                    rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
                    cov3D_precomp = None
                )
                
                rendered_sep = rendered_sep[:3, :, :]
        
            except:
                rendered_sep = torch.zeros_like(rendered_image).cuda()
        rendered_dy = None
        rendered_canon = None
        rendered_stat = None
        #rendered_depth = None
        rendered_motion = None
        rendered_flow_fwd = None
        rendered_flow_bwd = None
        if return_flow:
            
            focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
            focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
            tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
            viewmatrix = viewpoint_camera.world_view_transform.cuda()
            t = means3D * viewmatrix[0, :3]  + means3D * viewmatrix[1, :3] + means3D * viewmatrix[2, :3] + viewmatrix[3, :3]
            t = t.detach()
            
            # fwd flow
            flow_fwd = (means3D_right - means3D.detach()) * static_filter
            flow_fwd[:, 0] = flow_fwd[:, 0] * focal_x / t[:, 2]  + flow_fwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
            flow_fwd[:, 1] = flow_fwd[:, 1] * focal_y / t[:, 2]  + flow_fwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

            # bwd flow
            flow_bwd = (means3D_left - means3D.detach()) * static_filter
            flow_bwd[:, 0] = flow_bwd[:, 0] * focal_x / t[:, 2]  + flow_bwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
            flow_bwd[:, 1] = flow_bwd[:, 1] * focal_y / t[:, 2]  + flow_bwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

            rendered_flow_fwd, _, _ = rasterizer(
                means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                means2D = means2D.detach(),
                shs = None,
                colors_precomp = flow_fwd.repeat(1, NUM_CHANNELS//3),
                opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                cov3D_precomp = None)
            rendered_flow_bwd, _, _ = rasterizer(
                means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                means2D = means2D.detach(),
                shs = None,
                colors_precomp = flow_bwd.repeat(1, NUM_CHANNELS//3),
                opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                cov3D_precomp = None)
            
            rendered_flow_fwd = rendered_flow_fwd[:3, :, :]
            rendered_flow_bwd = rendered_flow_bwd[:3, :, :]

        #if return_depth:
        #    projected = viewpoint_camera.world_view_transform.T.unsqueeze(0) @ torch.cat([means3D * static_filter + pc.get_xyz * (1.-static_filter), torch.ones((screenspace_points.shape[0], 1), device="cuda")], dim=-1).unsqueeze(-1)
        #    projected = projected[:, 2:3, 0].repeat(1, 3)
        #    #assert False, [viewpoint_camera.world_view_transform, torch.cat([means3D_dy, pc.get_xyz_stat], dim=0)[0], projected.shape, projected[0]]
        #    if torch.any(bg_color==0):
        #        projected = 1./projected # because background is black, use disparity
        #    #assert False, [torch.max(projected), torch.min(projected)]
        #    rendered_depth, _, _ = rasterizer(
        #        means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
        #        means2D = means2D,
        #        shs = None,
        #        colors_precomp = projected,
        #        opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
        #        scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
        #        rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
        #        cov3D_precomp = None
        #    )
        #    if torch.any(bg_color==1):
        #        rendered_depth = 1./rendered_depth # rendered depth, use disparity
        if visualize:
            with torch.no_grad():
                is_dynamic = (static_filter == 1.).bool().view(-1)
                
                dy_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dy_dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dy_dir_pp_normalized = dy_dir_pp/dy_dir_pp.norm(dim=1, keepdim=True)
                dy_sh2rgb = eval_sh(pc.active_sh_degree, dy_view, dy_dir_pp_normalized)
                dy_precomp = torch.clamp_min(dy_sh2rgb + 0.5, 0.0).repeat(1, NUM_CHANNELS//3)
                
                static_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                static_dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                static_dir_pp_normalized = static_dir_pp/static_dir_pp.norm(dim=1, keepdim=True)
                static_sh2rgb = eval_sh(pc.active_sh_degree, static_view, static_dir_pp_normalized)
                static_precomp = torch.clamp_min(static_sh2rgb + 0.5, 0.0).repeat(1, NUM_CHANNELS//3)
                
                rendered_dy, _, _ = rasterizer(
                    means3D = means3D[is_dynamic],
                    means2D = means2D[is_dynamic],
                    shs = None,
                    colors_precomp = dy_precomp[is_dynamic],
                    opacities = opacity[is_dynamic],
                    scales = scales[is_dynamic],
                    rotations = rotations[is_dynamic],
                    cov3D_precomp = None
                )
                rendered_stat, _, _ = rasterizer(
                    means3D = pc.get_xyz[~is_dynamic],
                    means2D = means2D[~is_dynamic],
                    shs = None,
                    colors_precomp = static_precomp[~is_dynamic],
                    opacities = pc.get_opacity[~is_dynamic],
                    scales = pc.get_scaling[~is_dynamic],
                    rotations = pc.get_rotation[~is_dynamic],
                    cov3D_precomp = None
                )
                rendered_canon, _, _ = rasterizer(
                    means3D = pc.get_xyz[is_dynamic],
                    means2D = means2D[is_dynamic],
                    shs = None,
                    colors_precomp = static_precomp[is_dynamic],
                    opacities = pc.get_opacity[is_dynamic],
                    scales = pc.get_scaling[is_dynamic],
                    rotations = pc.get_rotation[is_dynamic],
                    cov3D_precomp = None
                )
                
                rendered_dy = rendered_dy[:3, :, :]
                rendered_stat = rendered_stat[:3, :, :]
                rendered_canon = rendered_canon[:3, :, :]

                


                #if not sf_reg:
                #    means3D_left, opacity_left, scales_left, rotations_left, dpos_left, dscale_left, drot_left = pc.get_deformed(time-motion_gap, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                #    means3D_right, opacity_right, scales_right, rotations_right, dpos_right, dscale_right, drot_right =  pc.get_deformed(time+motion_gap, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                #    mask = pc._isstatic[:, 0].bool()
                #
                #    scene_flow_left = means3D_left - means3D # from left to current
                #    scene_flow_right = means3D - means3D_right
                motion_per_point = (
                    torch.norm(scene_flow_left, dim=1) +
                    torch.norm(scene_flow_right, dim=1)
                )/2.
                #motion_per_point = motion_per_point * static_filter
                biggest_motion = torch.quantile(motion_per_point[mask], 0.9)
                smallest_motion = torch.quantile(motion_per_point[mask], 0.1)
                motion_per_point = (torch.clamp(motion_per_point, min=smallest_motion, max=biggest_motion)-smallest_motion)/(biggest_motion-smallest_motion+1e-6) 
                motion_per_point[~mask.detach()] *= 0.
                rendered_motion, _, _ = rasterizer(
                            means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
                            means2D = means2D,
                            shs = None,
                            colors_precomp = motion_per_point.view(-1, 1).repeat(1, NUM_CHANNELS),
                            opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
                            scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
                            rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
                            cov3D_precomp = None
                        )
                
                rendered_motion = rendered_motion[:3, :, :]


                if rendered_sep is None:
                    try:
                        rendered_sep, _, _ = rasterizer(
                            means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
                            means2D = means2D,
                            shs = None,
                            colors_precomp = static_filter.view(-1, 1).repeat(1, NUM_CHANNELS//3),
                            opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
                            scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
                            rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
                            cov3D_precomp = None
                        )
                        
                        rendered_sep = rendered_sep[:3, :, :]
                
                    except:
                        rendered_sep = torch.zeros_like(rendered_image).cuda()
                if rendered_flow_fwd is None:
                    focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
                    focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
                    tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
                    viewmatrix = viewpoint_camera.world_view_transform.cuda()
                    t = means3D * viewmatrix[0, :3]  + means3D * viewmatrix[1, :3] + means3D * viewmatrix[2, :3] + viewmatrix[3, :3]
                    t = t.detach()
                    
                    # fwd flow
                    flow_fwd = (means3D_right - means3D.detach())* static_filter
                    flow_fwd[:, 0] = flow_fwd[:, 0] * focal_x / t[:, 2]  + flow_fwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
                    flow_fwd[:, 1] = flow_fwd[:, 1] * focal_y / t[:, 2]  + flow_fwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

                    # bwd flow
                    flow_bwd = (means3D_left - means3D.detach()) * static_filter
                    flow_bwd[:, 0] = flow_bwd[:, 0] * focal_x / t[:, 2]  + flow_bwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
                    flow_bwd[:, 1] = flow_bwd[:, 1] * focal_y / t[:, 2]  + flow_bwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

                    rendered_flow_fwd, _, _ = rasterizer(
                        means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                        means2D = means2D.detach(),
                        shs = None,
                        colors_precomp = flow_fwd.repeat(1, NUM_CHANNELS//3),
                        opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                        scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                        rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                        cov3D_precomp = None)
                    rendered_flow_bwd, _, _ = rasterizer(
                        means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                        means2D = means2D.detach(),
                        shs = None,
                        colors_precomp = flow_bwd.repeat(1, NUM_CHANNELS//3),
                        opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                        scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                        rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                        cov3D_precomp = None)
                    
                    rendered_flow_fwd = rendered_flow_fwd[:3, :, :]
                    rendered_flow_bwd = rendered_flow_bwd[:3, :, :]
                #if rendered_depth is None:
                #    projected = viewpoint_camera.world_view_transform.T.unsqueeze(0) @ torch.cat([means3D * static_filter + pc.get_xyz * (1.-static_filter), torch.ones((screenspace_points.shape[0], 1), device="cuda")], dim=-1).unsqueeze(-1)
                #    projected = projected[:, 2:3, 0].repeat(1, 3)
                #    #assert False, [viewpoint_camera.world_view_transform, torch.cat([means3D_dy, pc.get_xyz_stat], dim=0)[0], projected.shape, projected[0]]
                #     if torch.any(bg_color==0): 
                #        projected = 1./projected # because background is black, use disparity
                #    #assert False, [torch.max(projected), torch.min(projected)]
                #    rendered_depth, _, _ = rasterizer(
                #        means3D = means3D * static_filter + pc.get_xyz * (1.-static_filter),
                #        means2D = means2D,
                #        shs = None,
                #        colors_precomp = projected,
                #        opacities = opacity * static_filter + pc.get_opacity * (1.-static_filter),
                #        scales = scales * static_filter + pc.get_scaling * (1.-static_filter),
                #        rotations = rotations * static_filter + pc.get_rotation * (1.-static_filter),
                #        cov3D_precomp = None
                #    )
                #    if torch.any(bg_color==1):
                #        rendered_depth = 1./rendered_depth # because background is white, use disparity
    else:
        #assert False, "motion visualization not supported yet"
        
        shs_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        dino_features = dino.squeeze()
        # clip_features = clip.squeeze()
        
        features_precomp = torch.cat((colors_precomp, dino_features), dim=1)
        # features_precomp = torch.cat((colors_precomp, dino_features, clip_features), dim=1)
        
        rendered_features, radii, rendered_depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = features_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image = rendered_features[:3, :, :]
        rendered_dino = rendered_features[3:6, :, :]
        # rendered_clip = rendered_features[6:-1, :, :]
        
        #scaling_tensor = scales.detach()
        if pc.ewa_prune:
            with torch.no_grad():
                scaling_tensor = torch.norm(scales, dim=-1, keepdim=True)
                opacity_tensor = opacity

        
        rendered_sep = None
        rendered_dy = None
        rendered_canon = None
        rendered_stat = None
        #rendered_depth = None
        rendered_motion = None
        rendered_flow_fwd = None
        rendered_flow_bwd = None
        #if return_depth:
        #    projected = viewpoint_camera.world_view_transform.T.unsqueeze(0) @ torch.cat([means3D, torch.ones((screenspace_points.shape[0], 1), device="cuda")], dim=-1).unsqueeze(-1)
        #    projected = projected[:, 2:3, 0].repeat(1, 3)
        #    if torch.any(bg_color==0):
        #        projected = 1./projected # because background is black, use disparity
        #    rendered_depth, _, _ = rasterizer(
        #        means3D = means3D,
        #        means2D = means2D,
        #        shs = None,
        #        colors_precomp = projected,
        #        opacities = opacity,
        #        scales = scales,
        #        rotations = rotations,
        #        cov3D_precomp = None
        #    )
        #    if torch.any(bg_color==1):
        #        rendered_depth = 1./rendered_depth # because background is black, use disparity
        if return_flow:
            focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
            focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
            tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
            viewmatrix = viewpoint_camera.world_view_transform.cuda()
            t = means3D * viewmatrix[0, :3]  + means3D * viewmatrix[1, :3] + means3D * viewmatrix[2, :3] + viewmatrix[3, :3]
            t = t.detach()
            
            # fwd flow
            flow_fwd = (means3D_right - means3D.detach())# * static_filter
            flow_fwd[:, 0] = flow_fwd[:, 0] * focal_x / t[:, 2]  + flow_fwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
            flow_fwd[:, 1] = flow_fwd[:, 1] * focal_y / t[:, 2]  + flow_fwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

            # bwd flow
            flow_bwd = (means3D_left - means3D.detach())# * static_filter
            flow_bwd[:, 0] = flow_bwd[:, 0] * focal_x / t[:, 2]  + flow_bwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
            flow_bwd[:, 1] = flow_bwd[:, 1] * focal_y / t[:, 2]  + flow_bwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

            rendered_flow_fwd, _, _ = rasterizer(
                means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                means2D = means2D.detach(),
                shs = None,
                colors_precomp = flow_fwd.repeat(1, NUM_CHANNELS//3),
                opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                cov3D_precomp = None)
            rendered_flow_bwd, _, _ = rasterizer(
                means3D = (means3D * static_filter + pc.get_xyz * (1.-static_filter)).detach(),
                means2D = means2D.detach(),
                shs = None,
                colors_precomp = flow_bwd.repeat(1, NUM_CHANNELS//3),
                opacities = (opacity * static_filter + pc.get_opacity * (1.-static_filter)).detach(),
                scales = (scales * static_filter + pc.get_scaling * (1.-static_filter)).detach(),
                rotations = (rotations * static_filter + pc.get_rotation * (1.-static_filter)).detach(),    
                cov3D_precomp = None)
            
            rendered_flow_fwd = rendered_flow_fwd[:3, :, :]
            rendered_flow_bwd = rendered_flow_bwd[:3, :, :]
        if visualize:
            rendered_canon, _, _ = rasterizer(
                means3D = pc.get_xyz,
                means2D = means2D,
                shs = pc.get_features,
                colors_precomp = None,
                opacities = pc.get_opacity,
                scales = pc.get_scaling,
                rotations = pc.get_rotation,
                cov3D_precomp = None
            )

            #if not sf_reg:
                #means3D_left, opacity_left, scales_left, rotations_left, dpos_left, dscale_left, drot_left = pc.get_deformed(time-motion_gap, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                #means3D_right, opacity_right, scales_right, rotations_right, dpos_right, dscale_right, drot_right =  pc.get_deformed(time+motion_gap, disable_offopa, disable_morph, multiply_offopa, anneal_band_iteration)
                #mask = pc._isstatic[:, 0].bool()
                
                #scene_flow_left = means3D_left - means3D # from left to current
                #scene_flow_right = means3D - means3D_right
            motion_per_point = (
                torch.norm(scene_flow_left, dim=1) +
                torch.norm(scene_flow_right, dim=1)
            )/2.
            #motion_per_point = motion_per_point * static_filter
            biggest_motion = torch.quantile(motion_per_point, 0.9)
            smallest_motion = torch.quantile(motion_per_point, 0.1)
            motion_per_point = (torch.clamp(motion_per_point, min=smallest_motion, max=biggest_motion)-smallest_motion)/(biggest_motion-smallest_motion+1e-6) 
            
            rendered_motion, _, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = motion_per_point.view(-1, 1).repeat(1, NUM_CHANNELS),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = None
            )
                

            if rendered_flow_fwd is None:
                focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
                focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
                tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
                viewmatrix = viewpoint_camera.world_view_transform.cuda()
                t = means3D * viewmatrix[0, :3]  + means3D * viewmatrix[1, :3] + means3D * viewmatrix[2, :3] + viewmatrix[3, :3]
                t = t.detach()
                
                # fwd flow
                flow_fwd = (means3D_right - means3D.detach()) #* static_filter
                flow_fwd[:, 0] = flow_fwd[:, 0] * focal_x / t[:, 2]  + flow_fwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
                flow_fwd[:, 1] = flow_fwd[:, 1] * focal_y / t[:, 2]  + flow_fwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

                # bwd flow
                flow_bwd = (means3D_left - means3D.detach()) #* static_filter
                flow_bwd[:, 0] = flow_bwd[:, 0] * focal_x / t[:, 2]  + flow_bwd[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
                flow_bwd[:, 1] = flow_bwd[:, 1] * focal_y / t[:, 2]  + flow_bwd[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

                rendered_flow_fwd, _, _ = rasterizer(
                    means3D = means3D.detach(),
                    means2D = means2D.detach(),
                    shs = None,
                    colors_precomp = flow_fwd.repeat(1, NUM_CHANNELS//3),
                    opacities = opacity.detach(),
                    scales = scales.detach(),
                    rotations = rotations.detach(),    
                    cov3D_precomp = None)
                rendered_flow_bwd, _, _ = rasterizer(
                    means3D = means3D.detach(),
                    means2D = means2D.detach(),
                    shs = None,
                    colors_precomp = flow_bwd.repeat(1, NUM_CHANNELS//3),
                    opacities = opacity.detach(),
                    scales = scales.detach(),
                    rotations = rotations.detach(),    
                    cov3D_precomp = None)
            #assert False, [viewpoint_camera.world_view_transform, torch.cat([means3D_dy, pc.get_xyz_stat], dim=0)[0], projected.shape, projected[0]]
            #if torch.any(bg_color==0):         
            #    projected = 1./projected # because background is black, use disparity
            #assert False, [torch.max(projected), torch.min(projected)]
            #if rendered_depth is None:
            #    projected = viewpoint_camera.world_view_transform.T.unsqueeze(0) @ torch.cat([means3D, torch.ones((screenspace_points.shape[0], 1), device="cuda")], dim=-1).unsqueeze(-1)
            #    projected = projected[:, 2:3, 0].repeat(1, 3)
            #    if torch.any(bg_color==0):
            #        projected = 1./projected # because background is black, use disparity
            #    rendered_depth, _, _ = rasterizer(
            #        means3D = means3D,
            #        means2D = means2D,
            #       shs = None,
            #        colors_precomp = projected,
            #        opacities = opacity,
            #        scales = scales,
            #        rotations = rotations,
            #        cov3D_precomp = None
            #    )
            #    if torch.any(bg_color==1):
            #        rendered_depth = 1./rendered_depth # because background is black, use disparity
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    #if motion_tensor is not None:
    #    left = torch.norm(motion_tensor, dim=1)
    #    #left = motions[known]
    #    print(float(torch.quantile(left, 0.9).detach()),
    #    float(torch.quantile(left, 0.8).detach()),
    #    float(torch.quantile(left, 0.7).detach()),
    #    float(torch.quantile(left, 0.5).detach()),
    #    float(torch.quantile(left, 0.3).detach()),
    #    float(torch.quantile(left, 0.2).detach()))

                
        
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            # store extra rendering
            "rendered_dino": rendered_dino,
            # "rendered_clip": rendered_clip,
            "rendered_sep": rendered_sep,
            "rendered_dy": rendered_dy,
            "rendered_canon": rendered_canon,
            "rendered_stat": rendered_stat,
            "rendered_depth": rendered_depth,
            "rendered_motion": rendered_motion,
            "rendered_flow_fwd": rendered_flow_fwd,
            "rendered_flow_bwd": rendered_flow_bwd,
            "sm_loss": sm_loss,
            "st_loss": st_loss,
            "motion_tensor": motion_tensor,
            "scaling_tensor": scaling_tensor,
            "opacity_tensor": opacity_tensor
        }