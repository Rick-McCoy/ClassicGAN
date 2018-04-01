from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pretty_midi as pm
import numpy as np
import pathlib
import random
import os
from tqdm import tqdm

CHANNEL_NUM = 6
CLASS_NUM = 72
INPUT_LENGTH = 384
BATCH_SIZE = 16
#0 Piano: 22584
#1 Chromatic Percussion: 216
#2 Orgam: 500
#3 Guitar: 6691
#4 Bass: 151
#5 Strings: 4356
#6 Ensemble: 5304
#7 Brass: 2044
#8 Reed: 2497
#9 Pipe: 1277
#10 Synth Lead: 70
#11 Synth Pad: 139
#12 Synth Effects: 51
#13 Ethnic: 91
#14 Percussive: 56
#15 Sound Effects: 36
#16 Drums: 411
def roll(path):
    try:
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
    except:
        print('Error while opening')
        raise Exception
    index = [0, 3, 5, 7, 8, 9]
    piano_rolls = [i.get_piano_roll()[24:96] for i in song.instruments]
    length = np.min([i.shape[1] for i in piano_rolls])
    if length < INPUT_LENGTH:
        print('Too short')
        raise Exception
    data = np.zeros(shape=(CHANNEL_NUM, CLASS_NUM, length))
    for piano_roll, instrument in zip(piano_rolls, song.instruments):
        if not instrument.is_drum:
            try:
                id = index.index(instrument.program // 8)
            except:
                continue
            data[id] = np.add(data[id], piano_roll[:, :length])
    if np.max(data) == 0:
        print('No notes')
        raise Exception
    data = data > 0
    data = (data - 0.5) * 2.0
    #while length < INPUT_LENGTH * BATCH_SIZE:
    #    np.concatenate((data, data), axis=-1)
    #    length *= 2
    data = np.split(data[:, :, :length // INPUT_LENGTH * INPUT_LENGTH], indices_or_sections=length // INPUT_LENGTH, axis=-1)
    return data

def build_dataset():
    pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    random.shuffle(pathlist)
    if not os.path.exists('Dataset'):
        os.mkdir('Dataset')
    cnt = 0
    concat = []
    for path in tqdm(pathlist):
        try:
            if not concat:
                concat = roll(str(path))
            else:
                concat.extend(roll(str(path)))
        except:
            tqdm.write(str(path) + ' ' + str(cnt))
            continue
        if len(concat) >= 1000:
            tqdm.write('saving %d' % cnt)
            np.save('Dataset/%d' % cnt, np.array(concat))
            concat = []
        cnt+=1

def main():
    build_dataset()

if __name__ == '__main__':
    main()
