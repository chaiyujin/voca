import os
import numpy as np
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

org_file = os.path.join(ROOT, 'training_data', 'processed_audio_deepspeech.pkl')
new_file = os.path.join(ROOT, 'training_data', 'processed_audio_deepspeech_avoffset_corrected.pkl')

if os.path.exists(new_file):
    # with open(new_file, 'rb') as fp:
    #     dat = pickle.load(fp, encoding='latin1')
    #     for obj in dat:
    #         print(obj)
    #         for seq in dat[obj]:
    #             print(' ', obj, seq, end=' ')
    #             print(dat[obj][seq]['audio'].shape, end=' ')
    #             print(dat[obj][seq]['sample_rate'])
    quit(0)

new_dat = dict()

with open(org_file, 'rb') as fp:
    dat = pickle.load(fp, encoding='latin1')
    for obj in dat:
        new_sequences = dict()
        print(obj)
        for seq in dat[obj]:
            print(' ', seq, end=' ')
            print(dat[obj][seq]['audio'].shape, end=' ')
            print(dat[obj][seq]['sample_rate'])
            new_sequences[seq] = dict(
                audio=dat[obj][seq]['audio'][6:, :, :],  # ! remove leading audio to reduce avoffset from 6 to 0
                sample_rate=dat[obj][seq]['sample_rate']
            )
        new_dat[obj] = new_sequences

with open(new_file, 'wb') as fp:
    pickle.dump(new_dat, fp)
