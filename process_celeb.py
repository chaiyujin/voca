import os
import argparse
import numpy as np
import pickle
import meshio
import librosa
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


def process_celeb(speaker, src_dir, tgt_dir, use_seqs):
    src_dir = os.path.expanduser(src_dir)
    tgt_dir = os.path.expanduser(tgt_dir)
    src_dir = os.path.join(src_dir, speaker)
    tgt_dir = os.path.join(tgt_dir, speaker)
    os.makedirs(tgt_dir, exist_ok=True)

    # source identity
    iden_path = os.path.join(src_dir, "fitted_identity/idle.obj")
    assert os.path.exists(iden_path)
    idle = meshio.load_mesh(iden_path)[0]
    with open(os.path.join(tgt_dir, "templates.pkl"), "wb") as fp:
        pickle.dump({speaker: idle}, fp)

    # find fitted videos
    tasks = []
    for cur_root, subdirs, _ in os.walk(os.path.join(src_dir, "fitted_video")):
        for subdir in subdirs:
            if subdir not in use_seqs:
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
        audio, sr = librosa.load(os.path.join(task, "audio.wav"), sr=16000)
        audio = (audio * 32767).astype(np.int16)
        raw_audios[subj][seq] = dict(audio=audio, sample_rate=sr)
        # load verts
        bin_list = glob(os.path.join(task, "meshes/frame_*.bin"))
        bin_list = sorted(bin_list)
        frames = []
        for bin_path in bin_list:
            verts_frame = np.memmap(bin_path, dtype="float32", mode="r").__array__()
            verts_frame = np.resize(verts_frame, (5023, 3))
            frames.append(verts_frame)
        frames = np.asarray(frames)
        frames = np.reshape(frames, (len(frames), 5023 * 3))
        frames = interpolate_features(frames, 30, 60)
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
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--source_dir", type=str, default="~/assets/CelebTalk/Processed")
    parser.add_argument("--target_dir", type=str, default="./training_data_celeb")
    parser.add_argument("--use_seqs", type=str, default="")
    args = parser.parse_args()

    process_celeb(args.speaker, args.source_dir, args.target_dir, args.use_seqs)
