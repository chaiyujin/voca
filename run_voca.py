"""
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
"""


import os
import cv2
import argparse
import numpy as np
import meshio
from glob import glob
from tqdm import tqdm
from utils.inference import inference
from template.selection import get_selection_obj, get_selection_triangles, get_selection_vidx
from yuki11.utils import VideoWriter, mesh_viewer


PART = "face1"
vidx = get_selection_vidx(PART)
obj_path = get_selection_obj(PART)
mesh_viewer.set_template(obj_path)
mesh_viewer.set_shading_mode("smooth")


def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ["true", "t", "yes", "y"]:
            return True
        elif val.lower() in ["false", "f", "no", "n"]:
            return False
    return False


def render_results(res_dir):
    vpath = os.path.join(res_dir, "render.mp4")
    if os.path.exists(vpath):
        return
    apath = os.path.join(res_dir, "audio.wav")
    obj_files = sorted(glob(os.path.join(res_dir, "meshes/*.obj")))

    writer = VideoWriter(vpath, fps=60, src_audio_path=apath, high_quality=True)
    for obj_file in tqdm(obj_files):
        vert = meshio.load_mesh(obj_file)[0]
        vert = vert[vidx]
        vert = vert * 1.4
        vert[:, 1] += 0.02
        im = mesh_viewer.render_verts(vert)[:, :, [2, 1, 0]]
        writer.write(im)
        # cv2.imshow("im", im)
        # cv2.waitKey(33)
    writer.release()


parser = argparse.ArgumentParser(description="Voice operated character animation")
parser.add_argument("--tf_model_fname", default="./model/gstep_52280.model", help="Path to trained VOCA model")
parser.add_argument("--ds_fname", default="./ds_graph/output_graph.pb", help="Path to trained DeepSpeech model")
parser.add_argument("--audio_fname", default="./audio/test_sentence.wav", help="Path of input speech sequence")
parser.add_argument(
    "--template_fname",
    default="./template/FLAME_sample.ply",
    help='Path of "zero pose" template mesh in" FLAME topology to be animated',
)
parser.add_argument("--condition_idx", type=int, default=3, help="Subject condition id in [1,8]")
parser.add_argument("--uv_template_fname", default="", help="Path of a FLAME template with UV coordinates")
parser.add_argument("--texture_img_fname", default="", help="Path of the texture image")
parser.add_argument("--out_path", default="./voca/animation_output", help="Output path")
parser.add_argument("--visualize", default="True", help="Visualize animation")

args = parser.parse_args()

tf_model_fname = args.tf_model_fname
ds_fname = args.ds_fname
audio_fname = args.audio_fname
template_fname = args.template_fname
condition_idx = args.condition_idx
out_path = args.out_path

uv_template_fname = args.uv_template_fname
texture_img_fname = args.texture_img_fname

if not os.path.exists(out_path):
    os.makedirs(out_path)

done_flag = os.path.join(out_path, "meshes", "done.lock")
if not os.path.exists(done_flag):
    inference(
        tf_model_fname,
        ds_fname,
        audio_fname,
        template_fname,
        condition_idx,
        out_path,
        str2bool(args.visualize),
        uv_template_fname=uv_template_fname,
        texture_img_fname=texture_img_fname,
    )
    with open(done_flag, "w") as fp:
        fp.write("")

render_results(out_path)
