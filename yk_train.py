'''
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
'''

import os
import shutil
import argparse
import configparser
import tensorflow as tf

from config_parser import read_config, create_default_config
from utils.data_handler_celeb import DataHandler
from utils.batcher import Batcher
from utils.voca_model import VOCAModel as Model


def main(exp_dir, epoch):
    # Prior to training, please adapt the hyper parameters in the config_parser.py and run the script to generate
    # the training config file use to train your own VOCA model.
    assert os.path.exists(exp_dir)

    init_config_fname = os.path.join(exp_dir, 'training_config.cfg')
    if not os.path.exists(init_config_fname):
        print('Config not found %s' % init_config_fname)
        create_default_config(init_config_fname)

    config = configparser.ConfigParser()
    config.read(init_config_fname)

    # Path to cache the processed audio
    config.set('Input Output', 'processed_audio_path', './training_data/processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type'))
    config.set('Input Output', 'checkpoint_dir', os.path.join(exp_dir, 'training'))
    # celeb related
    config.set('Input Output', 'celeb',                       os.path.basename(exp_dir))
    config.set('Input Output', 'celeb_verts_mmaps_path',      os.path.join(exp_dir, 'data', 'train', 'data_verts.npy'))
    config.set('Input Output', 'celeb_raw_audio_path',        os.path.join(exp_dir, 'data', 'train', 'raw_audio.pkl'))
    config.set('Input Output', 'celeb_templates_path',        os.path.join(exp_dir, 'data', 'train', 'templates.pkl'))
    config.set('Input Output', 'celeb_data2array_verts_path', os.path.join(exp_dir, 'data', 'train', 'subj_seq_to_idx.pkl'))
    config.set('Input Output', 'celeb_processed_audio_path',  os.path.join(exp_dir, 'data', 'train', 'processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type')))
    # epoch
    config.set('Learning Parameters', 'epoch_num', str(epoch))

    checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
    if os.path.exists(checkpoint_dir):
        print('Checkpoint dir already exists %s. Try to delete.' % checkpoint_dir)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_fname = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(config_fname):
        print('Use existing config %s' % config_fname)
    else:
        with open(config_fname, 'w') as fp:
            config.write(fp)
            fp.close()

    config = read_config(config_fname)
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    with tf.Session() as session:
        model = Model(session=session, config=config, batcher=batcher)
        model.build_graph()
        model.load()
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()

    main(args.exp_dir, args.epoch)
