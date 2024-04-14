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

import numpy as np
import torch
import trimesh
from lib.common.imutils import load_MODNet, process_image
from lib.common.render import Render
from lib.common.train_util import Format
from lib.dataset.mesh_util import SMPLX
from utils.body_utils.lib import smplx
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.pixielib.pixie import PIXIE
from lib.pixielib.utils.config import cfg as pixie_cfg
from PIL import ImageFile
from termcolor import colored
from torchvision.models import detection

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset:
    def __init__(self, cfg, device, no_crop=False):

        self.image_path = cfg["image_path"]
        self.hps_type = cfg["hps_type"]
        self.smpl_type = "smplx"
        self.no_crop = no_crop
        self.device = device

        self.cameras = np.load("./data/PuzzleIOI/camera.npy", allow_pickle=True).item()

        # smpl related
        self.smpl_data = SMPLX()

        if not self.no_crop:

            if self.hps_type == "pixie":
                self.hps = PIXIE(config=pixie_cfg, device=self.device)

            print(
                colored(
                    f"SMPL-X estimate with {Format.start} {self.hps_type.upper()} {Format.end}",
                    "green"
                )
            )

            self.detector = detection.maskrcnn_resnet50_fpn(
                weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights
            )
            self.detector.eval()
            self.smpl_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)

        else:
            self.detector = None
            self.smpl_model = None

            print(colored(f"SMPL-X from {Format.start} PuzzleIOI fitting {Format.end}", "green"))

        self.modnet = load_MODNet(
            "thirdparties/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt"
        ).to(self.device)

        self.render = Render(size=512, device=self.device)

    def __len__(self):
        return 1

    def __getitem__(self, index):

        img_path = self.image_path
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        arr_dict = process_image(img_path, 1024, self.detector, modnet=self.modnet)
        arr_dict.update({"name": img_name})

        if self.no_crop:
            pkl_path = img_path.replace("masked", "smplx").replace("07_C.jpg", "smplx.pkl")
            scan_path = img_path.replace("masked/07_C.jpg", "scan.obj")
            preds_dict = np.load(pkl_path, allow_pickle=True)

            self.smpl_model = smplx.create(
                self.smpl_data.model_dir,
                model_type='smplx',
                gender=preds_dict["gender"],
                age="adult",
                use_face_contour=False,
                use_pca=True,
                num_betas=10,
                num_expression_coeffs=10,
                flat_hand_mean=False,
                ext='pkl'
            ).to(self.device)

            self.scan = trimesh.load_mesh(scan_path)
            self.center = (self.scan.vertices.mean(0) / 1000.0).astype(np.float32)

        else:
            with torch.no_grad():
                if self.hps_type == "pixie":
                    preds_dict = self.hps.forward(arr_dict["img_hps"].to(self.device))
                else:
                    raise NotImplementedError

        arr_dict["smpl_faces"] = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(
                self.device
            )
        )
        arr_dict["type"] = self.smpl_type

        if self.no_crop:
            
            arr_dict["transl"] = torch.as_tensor([preds_dict["transl"]]).to(self.device).float()

            for key in ["gender", "keypoints_3d", "transl"]:
                preds_dict.pop(key, None)

            for key in preds_dict.keys():
                preds_dict[key] = torch.as_tensor(preds_dict[key]).to(self.device).float()

            arr_dict.update(preds_dict)

            smplx_obj = self.smpl_model(**preds_dict)
            arr_dict["smpl_verts"] = smplx_obj.vertices
            arr_dict["smpl_joints"] = smplx_obj.joints

            arr_dict["scale"] = torch.tensor([0.6600]).unsqueeze(1).to(self.device).float()
            arr_dict["transl"] += (
                torch.tensor([[self.center]]) + torch.tensor([[[-0.06, -0.40, 0.0]]])
            ).to(self.device)
        else:
            if self.hps_type == "pixie":
                arr_dict.update(preds_dict)
                arr_dict["global_orient"] = preds_dict["global_pose"]
                arr_dict["betas"] = preds_dict["shape"]    #200
                arr_dict["smpl_verts"] = preds_dict["vertices"]

                scale, tranX, tranY = preds_dict["cam"].split(1, dim=1)
                # 1.1435, 0.0128, 0.3520

            arr_dict["scale"] = scale.unsqueeze(1)
            arr_dict["transl"] = (
                torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                          dim=1).unsqueeze(1).to(self.device).float()
            )

            # from rot_mat to rot_6d for better optimization
            N_body, N_pose = arr_dict["body_pose"].shape[:2]
            arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
            arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(
                N_body, 1, -1
            )

        return arr_dict

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")
