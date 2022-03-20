import os
import json
import argparse
import numpy as np
import pickle
import meshio
import librosa
from shutil import copyfile
from tqdm import tqdm
from glob import glob


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             features[:, feat])
    return output_features


def process_celeb(speaker, src_dir, dat_dir, avoffset_ms):
    src_dir = os.path.expanduser(src_dir)

    src_dir = os.path.join(src_dir, speaker)
    tgt_dir = os.path.join(dat_dir, "train")
    os.makedirs(tgt_dir, exist_ok=True)

    # source identity
    iden_path = os.path.join(src_dir, "fitted/identity/identity.obj")
    assert os.path.exists(iden_path)
    idle, tris, _ = meshio.load_mesh(iden_path)
    meshio.save_ply(os.path.join(os.path.dirname(dat_dir), "template.ply"), idle, tris)
    with open(os.path.join(tgt_dir, "templates.pkl"), "wb") as fp:
        pickle.dump({speaker: idle}, fp)

    # find fitted videos
    tasks = []
    for cur_root, subdirs, _ in os.walk(os.path.join(src_dir, "fitted")):
        for subdir in subdirs:
            if not subdir.startswith("trn"):
                continue
            assert subdir.startswith("trn")
            tasks.append(os.path.join(cur_root, subdir))
        break
    print(tasks)

    # load and cache
    face_verts = []
    raw_audios = {speaker: {}}
    subj_seq_to_idx = {speaker: {os.path.basename(task): {} for task in tasks}}
    for task in tqdm(tasks):
        subj = speaker
        seq = os.path.basename(task)
        # load audio
        ss = task.split('/')
        apath = os.path.join('/'.join(ss[:-2]), ss[-1], "audio.wav")
        copyfile(apath, os.path.join(tgt_dir, f"{ss[-1]}.wav"))
        audio, sr = librosa.load(apath, sr=16000)
        audio = (audio * 32767).astype(np.int16)
        raw_audios[subj][seq] = dict(audio=audio, sample_rate=sr)
        # load verts
        npy_list = glob(os.path.join(task, "meshes/*.npy"))
        frames = [np.load(x) for x in sorted(npy_list)]
        frames = np.asarray(frames)
        # load fps
        info_path = os.path.join('/'.join(ss[:-2]), ss[-1], "info.json")
        with open(info_path) as fp:
            data_info = json.load(fp)
            fps = data_info['fps']
        # handle avoffset
        if avoffset_ms is not None:
            assert avoffset_ms == data_info['avoffset_ms']
            tqdm.write("> Handling avoffset_ms: {}".format(avoffset_ms))
            avoffset = int(np.round(fps * avoffset_ms / 1000.0))
            if avoffset > 0:  # audio ts larger than video, so delay video
                padding = frames[:1].repeat(avoffset, axis=0)
                frames = np.concatenate((padding, frames), axis=0)
            else:  # audio ts smaller than video, so clip video
                frames = frames[avoffset:]
        # upsample to 60 fps
        frames = np.reshape(frames, (len(frames), 5023 * 3))
        frames = interpolate_features(frames, fps, 60)
        frames = np.reshape(frames, (len(frames), 5023, 3)).astype(np.float32)

        # append new frames
        for ifrm, verts in enumerate(frames):
            idx = len(face_verts)
            face_verts.append(verts)
            subj_seq_to_idx[subj][seq][ifrm] = idx

    with open(os.path.join(tgt_dir, "raw_audio.pkl"), "wb") as fp:
        pickle.dump(raw_audios, fp)
    with open(os.path.join(tgt_dir, "subj_seq_to_idx.pkl"), "wb") as fp:
        pickle.dump(subj_seq_to_idx, fp)
    np.save(os.path.join(tgt_dir, "data_verts.npy"), face_verts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_src", type=str, required=True)
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--avoffset_ms", type=float)
    parser.add_argument("--source_dir", type=str, default="~/Documents/Project2021/stylized-sa/data/datasets/talk_video/{}/data")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.format(args.data_src)
    process_celeb(args.speaker, source_dir, args.data_dir, args.avoffset_ms)
