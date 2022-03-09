import os
import cv2
import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm, trange
from meshio import load_mesh
from template.selection import get_selection_obj, get_selection_triangles, get_selection_vidx
from template.selection import LIPS_VIDX, FACE_NEYE_VIDX, FACE_LOWER_VIDX
from yk_metrics import verts_dist

from yuki11.utils import mesh_viewer

VIEW_VIDX = get_selection_vidx('face1')
mesh_viewer.set_template(get_selection_obj('face1'))
mesh_viewer.set_shading_mode('smooth')


def load_res(res_dir):
    ply_files = glob(os.path.join(res_dir, "*.ply"))
    ply_files = sorted(ply_files)
    frames = []
    for fpath in tqdm(ply_files, desc="load_res", leave=False):
        verts, _, _ = load_mesh(fpath)
        frames.append(verts)
    return np.asarray(frames, dtype=np.float32)


def load_dat(dat_dir):
    npy_files = glob(os.path.join(dat_dir, "*.npy"))
    npy_files = sorted(npy_files)
    frames = []
    for fpath in tqdm(npy_files, desc="load_dat", leave=False):
        frames.append(np.load(fpath))
    return np.asarray(frames, dtype=np.float32)


def _dat_dir(data_src, speaker, seq_id):
    return "../../stylized-sa/data/datasets/talk_video/{}/data/{}/fitted/{}/meshes".format(data_src, speaker, seq_id)


def _res_dir(data_src, speaker, seq_id):
    return "yk_exp/{}/{}/results/clip-{}/meshes".format(data_src, speaker, seq_id)


def _spk_dir(data_src, speaker):
    return "yk_exp/{}/{}/results".format(data_src, speaker)


logs = []

data_src = "celebtalk"
speaker = "m001_trump"
for seq_id in ['vld-000', 'vld-001']:
    preds = load_res(_res_dir(data_src, speaker, seq_id))
    reals = load_dat(_dat_dir(data_src, speaker, seq_id))
    n_frames = min(len(preds), len(reals))
    for i in trange(n_frames):
        pred = torch.tensor(preds[i].copy())
        REAL = torch.tensor(reals[i].copy())

        log = dict()
        log["mvd-avg:lips" ] = verts_dist(pred, REAL, LIPS_VIDX, reduction='mean')
        log["mvd-max:lips" ] = verts_dist(pred, REAL, LIPS_VIDX, reduction='max')
        log["mvd-avg:lower"] = verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='mean')
        log["mvd-max:lower"] = verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='max')
        log["mvd-avg:face" ] = verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='mean')
        log["mvd-max:face" ] = verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='max')
        for key in log:
            log[key] = log[key].item()
        logs.append(log)

        im0 = mesh_viewer.render_verts(pred[VIEW_VIDX])
        im1 = mesh_viewer.render_verts(REAL[VIEW_VIDX])
        cv2.imshow('img', np.concatenate((im0, im1), axis=1))
        cv2.waitKey(1)

metrics = {}
for log in logs:
    for k, v in log.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)
with open(os.path.join(_spk_dir(data_src, speaker), "metrics.json"), "w") as fp:
    json.dump(metrics, fp)
