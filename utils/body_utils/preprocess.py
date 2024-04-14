# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
import os

import numpy as np
import torch
import torchvision
import trimesh
from lib.common.config import cfg
from lib.common.imutils import blend_rgb_norm
from lib.common.train_util import Format
from lib.dataset.mesh_util import *
from lib.dataset.TestDataset import TestDataset
from lib.Normal import Normal
from termcolor import colored

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-in_path", "--in_path", type=str, default=None)
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument(
        "-cfg", "--config", type=str, default="./utils/body_utils/configs/body.yaml"
    )
    parser.add_argument("-novis", action="store_true")
    parser.add_argument("-nocrop", action="store_true")

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)

    device = torch.device(f"cuda:{args.gpu_device}")

    # setting for testing on in-the-wild images
    cfg_show_list = [
        "test_gpus",
        [args.gpu_device],
        "mcube_res",
        512,
        "clean_mesh",
        True,
        "test_mode",
        True,
        "batch_size",
        1,
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    # load normal model
    normal_net = Normal.load_from_checkpoint(
        cfg=cfg, checkpoint_path=cfg.normal_path, map_location=device, strict=False
    )
    normal_net = normal_net.to(device)
    normal_net.netG.eval()
    print(
        colored(
            f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"
        )
    )

    dataset_param = {
        "image_path": os.path.join("./data", args.in_path, "masked/07_C.jpg"),
        "hps_type": "pixie",
    }

    dataset = TestDataset(dataset_param, device, no_crop=args.nocrop)

    data = dataset[0]

    os.makedirs(osp.join(args.out_dir, "png"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "normal"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "vis"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "obj"), exist_ok=True)

    in_tensor = {
        "smpl_faces": data["smpl_faces"],
        "image": data["img_icon"].to(device),
        "mask": data["img_mask"].to(device),
    }

    smpl_verts = (data["smpl_verts"] + data["transl"]) * data["scale"]
    data["smpl_joints"] = (data["smpl_joints"] + data["transl"]) * data["scale"]
    data["smpl_joints"] *= torch.tensor([1.0, -1.0, -1.0]).to(device)

    # render optimized mesh as normal [-1,1]
    in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
        smpl_verts.to(device),
        in_tensor["smpl_faces"],
    )

    T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

    with torch.no_grad():
        in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

    # silhouette loss
    smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
    gt_arr = in_tensor["mask"].repeat(1, 1, 2)
    diff_S = torch.abs(smpl_arr - gt_arr)

    per_loop_lst = []

    per_loop_lst.extend([
        in_tensor["image"],
        in_tensor["T_normal_F"],
        in_tensor["normal_F"],
        diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
    ])
    per_loop_lst.extend([
        in_tensor["image"],
        in_tensor["T_normal_B"],
        in_tensor["normal_B"],
        diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
    ])

    if not args.novis:

        get_optim_grid_image(per_loop_lst, None, nrow=2, type="smpl").save(
            osp.join(args.out_dir, f"vis/{data['name']}_smpl.png")
        )
        img_crop_path = osp.join(args.out_dir, "png", f"{data['name']}_crop.png")
        torchvision.utils.save_image(data["img_crop"], img_crop_path)
        img_normal_F_path = osp.join(args.out_dir, "normal", f"{data['name']}_normal_front.png")
        img_normal_B_path = osp.join(args.out_dir, "normal", f"{data['name']}_normal_back.png")
        normal_F = in_tensor['normal_F'].detach().cpu()
        normal_F_mask = (normal_F.abs().sum(1) > 1e-6).to(normal_F)
        normal_B = in_tensor['normal_B'].detach().cpu()
        normal_B_mask = (normal_B.abs().sum(1) > 1e-6).to(normal_B)
        torchvision.utils.save_image(
            torch.cat([(normal_F + 1.0) * 0.5, normal_F_mask.unsqueeze(1)], dim=1),
            img_normal_F_path
        )

        torchvision.utils.save_image(
            torch.cat([(normal_B + 1.0) * 0.5, normal_B_mask.unsqueeze(1)], dim=1),
            img_normal_B_path
        )

        rgb_norm_F = blend_rgb_norm(in_tensor["normal_F"], data)
        rgb_norm_B = blend_rgb_norm(in_tensor["normal_B"], data)
        rgb_T_norm_F = blend_rgb_norm(in_tensor["T_normal_F"], data)
        rgb_T_norm_B = blend_rgb_norm(in_tensor["T_normal_B"], data)

        img_overlap_path = osp.join(args.out_dir, f"vis/{data['name']}_overlap.png")
        torchvision.utils.save_image(
            torch.cat([data["img_raw"], rgb_norm_F, rgb_norm_B], dim=-1) / 255., img_overlap_path
        )

        smpl_overlap_path = osp.join(args.out_dir, f"vis/{data['name']}_smpl_overlap.png")
        torchvision.utils.save_image((data["img_raw"] + rgb_T_norm_F) / 2. / 255.,
                                     smpl_overlap_path)

    smpl_obj_lst = []

    smpl_obj = trimesh.Trimesh(
        smpl_verts[0].detach().cpu(),
        in_tensor["smpl_faces"][0].detach().cpu(),
        process=False,
        maintains_order=True,
    )

    smpl_obj_path = f"{args.out_dir}/obj/{data['name']}_smpl.obj"

    smpl_obj.export(smpl_obj_path)
    smpl_info = {}
    for key in [
        "betas",
        "body_pose",
        "global_orient",
        "transl",
        "expression",
        "jaw_pose",
        "left_hand_pose",
        "right_hand_pose",
        "scale",
        "smpl_joints",
    ]:
        smpl_info[key] = data[key].cpu()
        
    smpl_info["transl"] = smpl_info["transl"].squeeze(0)

    np.save(
        smpl_obj_path.replace(".obj", ".npy"),
        smpl_info,
        allow_pickle=True,
    )
