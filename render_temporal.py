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
from scene.temporal_scene import TemporalScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.temporal_render import temporal_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments.temporal import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer.temporal_render import TemporalGaussianModel
from utils.image_utils import psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure
import imutils

@torch.no_grad()
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depths_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    canons_path = os.path.join(model_path, name, "ours_{}".format(iteration), "canon")
    if gaussians.enable_static:
        seps_path = os.path.join(model_path, name, "ours_{}".format(iteration), "sep")
        dys_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dys")
        stats_path = os.path.join(model_path, name, "ours_{}".format(iteration), "stats")

    #print(disable_offopa)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depths_path, exist_ok=True)
    makedirs(canons_path, exist_ok=True)
    if gaussians.enable_static:
        makedirs(seps_path, exist_ok=True)
        makedirs(dys_path, exist_ok=True)
        makedirs(stats_path, exist_ok=True)
    
    psnrs = []
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    ssims = []
    #preds = []
    #target = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #print(disable_offopa, multiply_offopa)
        start.record()
        temporal_render(view, gaussians, pipeline, background, disable_offscale=disable_offscale, disable_offopa=disable_offopa, disable_morph=disable_morph, multiply_offopa=multiply_offopa,
                                    disable_deform=False, 
                                    return_sep=False,
                                    return_depth=False,
                                    visualize=False,)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        render_pkg = temporal_render(view, gaussians, pipeline, background, disable_offscale=disable_offscale, disable_offopa=disable_offopa, disable_morph=disable_morph, multiply_offopa=multiply_offopa,
                                    disable_deform=False, 
                                    return_sep=True,
                                    return_depth=True,
                                    visualize=True)
        rendering = render_pkg["render"] # must have
        depth = render_pkg["rendered_depth"] # must have
        depth = imutils.np2png_d( [depth[0, ...].cpu().numpy()], None, colormap="jet")
        depth = torch.from_numpy(depth).permute(2, 0, 1)
        sep = render_pkg["rendered_sep"]
        dy = render_pkg["rendered_dy"]
        canon = render_pkg["rendered_canon"] # must have
        stat = render_pkg["rendered_stat"]

        gt = view.original_image[0:3, :, :]
        psnrs.append(float(psnr(rendering, gt).mean().detach().double()))
        #preds.append(rendering.detach().double())
        #target.append(gt.detach().double())
        ssims.append(ssim(rendering[None], gt[None]))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depths_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(canon, os.path.join(canons_path, '{0:05d}'.format(idx) + ".png"))
        if gaussians.enable_static:
            torchvision.utils.save_image(sep, os.path.join(seps_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(stat, os.path.join(stats_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(dy, os.path.join(dys_path, '{0:05d}'.format(idx) + ".png"))
            
    
    print(f"PSNR for {name}: {sum(psnrs)/float(len(psnrs))}")
    print(f"SSIM for {name}: {sum(ssims)/float(len(ssims))}")
    print(f"Render for {name}: {sum(times)/float(len(times))}")

@torch.no_grad()
def render_set_viewonly(model_path, name, iteration, views, gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    #gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    #print(disable_offopa)
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    #makedirs(gts_path, exist_ok=True)
    #psnrs = []
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    #ssims = []
    #preds = []
    #target = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #print(disable_offopa, multiply_offopa)
        render_pkg = temporal_render(view, gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa,
        disable_deform=False, 
                                    return_sep=True,
                                    return_depth=True,
                                    visualize=True)
        rendering = render_pkg["render"]
        depth = render_pkg["rendered_depth"]
        depth = imutils.np2png_d( [depth[0, ...].cpu().numpy()], None, colormap="jet")
        depth = torch.from_numpy(1.-depth).permute(2, 0, 1)
                    
        #gt = view.original_image[0:3, :, :]
        #psnrs.append(float(psnr(rendering, gt).mean().detach().double()))
        #preds.append(rendering.detach().double())
        #target.append(gt.detach().double())
        #ssims.append(ssim(rendering[None], gt[None]))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    #print(f"PSNR for {name}: {sum(psnrs)/float(len(psnrs))}")
    #print(f"SSIM for {name}: {sum(ssims)/float(len(ssims))}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_360: bool, disable_offscale: bool, disable_offopa: bool, disable_morph: bool, multiply_offopa: bool, enable_offsh: bool, separate_offopa: bool, separate_offsh: bool, enable_static: bool,
                stop_gradient: bool, use_skips: bool,
                new_deform: bool, shrink_lr: bool,
                use_nte: bool, use_SE: bool,
                mult_quaternion: bool,
                rotate_sh: bool,
                dynamic_sep: bool,
                use_ResFields: bool,
                ewa_prune: bool):
    with torch.no_grad():
        gaussians = TemporalGaussianModel(dataset.sh_degree, enable_offsh=enable_offsh, separate_offopa=separate_offopa, separate_offsh=separate_offsh, enable_static=enable_static, init_mode_gaussian=False,
            stop_gradient=stop_gradient, use_skips=use_skips,
            new_deform=new_deform, shrink_lr=shrink_lr, use_nte=use_nte, use_SE=use_SE,
            anneal_band=False, anneal_band_time=False, anneal_band_steps=None,
            mult_quaternion=mult_quaternion, rotate_sh=rotate_sh,
            posbase_pe=dataset.posbase_pe, timebase_pe=dataset.timebase_pe,
            defor_depth=dataset.defor_depth, net_width=dataset.net_width,
            dynamic_sep=dynamic_sep, use_ResFields=use_ResFields, ResField_mode=dataset.ResField_mode,
            capacity=dataset.capacity, ewa_prune=ewa_prune)
        scene = TemporalScene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        #dataset.white_background = white_background
        #print("Don't believe in PSNR if using white background!")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_360:
            render_set_viewonly(dataset.model_path, "360", scene.loaded_iter, scene.getOrbitCameras(), gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa)


        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, disable_offscale, disable_offopa, disable_morph, multiply_offopa)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_360", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_offscale", action="store_true")
    parser.add_argument("--disable_offopa", action="store_true")
    parser.add_argument("--disable_morph", action="store_true")
    parser.add_argument("--multiply_offopa", action="store_true")
    parser.add_argument("--enable_offsh", action="store_true")
    parser.add_argument("--separate_offopa", action="store_true")
    parser.add_argument("--separate_offsh", action="store_true")
    parser.add_argument("--enable_static", action="store_true")
    parser.add_argument("--stop_gradient", action="store_true")
    parser.add_argument("--use_skips", action="store_true")
    parser.add_argument("--new_deform", action="store_true")
    parser.add_argument("--shrink_lr", action="store_true")
    parser.add_argument("--use_nte", action="store_true")
    parser.add_argument("--use_SE", action="store_true")
    parser.add_argument("--mult_quaternion", action="store_true")
    parser.add_argument("--rotate_sh", action="store_true")
    parser.add_argument("--dynamic_sep", action="store_true")
    parser.add_argument("--use_ResFields", action="store_true")
    parser.add_argument("--ewa_prune", action="store_true")
    
    args = get_combined_args(parser)
    #assert False, args.disable_offopa
    args.eval = True
    print("Rendering " + args.model_path)

    assert not args.dynamic_sep, "Not supported for now"
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.separate_offopa:
        assert not args.disable_offopa and not args.multiply_offopa, "Already using separate offopa network!"

    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_360, 
    args.disable_offscale, args.disable_offopa, args.disable_morph, args.multiply_offopa, args.enable_offsh, args.separate_offopa, args.separate_offsh,
    args.enable_static, args.stop_gradient, args.use_skips, args.new_deform, args.shrink_lr, args.use_nte, args.use_SE,
    args.mult_quaternion, args.rotate_sh, args.dynamic_sep, args.use_ResFields, args.ewa_prune)
    