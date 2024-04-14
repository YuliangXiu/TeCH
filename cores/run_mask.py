import matplotlib.pyplot as plt
import pyrender
import trimesh
import numpy as np
from tqdm import tqdm
import os

import multiprocessing as mp
from functools import partial
from multiprocessing import Pool

scene = pyrender.Scene()
light = pyrender.SpotLight(
    color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
)

cameras = np.load("./data/PuzzleIOI/camera.npy", allow_pickle=True).item()


def render(root_dir):

    scan_file = os.path.join(root_dir, "scan.obj")
    ref_img_file = os.path.join(root_dir, "images/07_C.jpg")
    camera_cali = cameras["07_C"]

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )
    camera_pose = camera_cali['extrinsic']
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh = trimesh.intersections.slice_mesh_plane(scan_mesh, [0, 1, 0], [0, -580.0, 0])
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(ref_img.shape[1], ref_img.shape[0])
    color, _ = r.render(scene)
    mask = (color == color[0, 0]).sum(axis=2, keepdims=True) != 3
    masked_img = ref_img * mask

    scene.clear()

    out_dir = os.path.join(root_dir, "masked")
    os.makedirs(out_dir, exist_ok=True)

    plt.imsave(os.path.join(out_dir, "07_C.jpg"), masked_img)

    return masked_img


if __name__ == "__main__":
    
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cameras = np.load("./data/PuzzleIOI/camera.npy", allow_pickle=True).item()

    subjects = np.loadtxt("all_subjects.txt", dtype=str, delimiter=" ")[:, 0]
    subjects = [f"./data/{outfit}/" for outfit in subjects]

    print("CPU:", mp.cpu_count())
    print("propress", len(subjects))

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    render,
                ), subjects
            ),
            total=len(subjects)
        ):
            pass

    pool.close()
    pool.join()
    
    print('Finish Rendering.')
