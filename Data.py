from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pretty_midi as pm
import numpy as np

CHANNEL_NUM = 7
CLASS_NUM = 72
INPUT_LENGTH = 384
BATCH_NUM = 32
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
        raise Exception
    length = np.min([i.get_piano_roll().shape[1] for i in song.instruments])
    if length == 0:
        raise Exception
    length = length if length < INPUT_LENGTH * BATCH_NUM else INPUT_LENGTH * BATCH_NUM
    data = np.zeros(shape=(CHANNEL_NUM, CLASS_NUM, length))
    for i in song.instruments:
        if not i.is_drum:
            if i.program // 8 == 0:
                data[0] = np.add(data[0], i.get_piano_roll()[24:96, :length])
            elif i.program // 8 == 3:
                data[1] = np.add(data[1], i.get_piano_roll()[24:96, :length])
            elif i.program // 8 == 5:
                data[2] = np.add(data[2], i.get_piano_roll()[24:96, :length])
            elif i.program // 8 == 7:
                data[3] = np.add(data[3], i.get_piano_roll()[24:96, :length])
            elif i.program // 8 == 8:
                data[4] = np.add(data[4], i.get_piano_roll()[24:96, :length])
            elif i.program // 8 == 9:
                data[5] = np.add(data[5], i.get_piano_roll()[24:96, :length])
            else:
                data[6] = np.add(data[6], i.get_piano_roll()[24:96, :length])
    if np.max(data) == 0:
        raise Exception
    data = data > 0
    data = (data - 0.5) * 2.0
    while length < INPUT_LENGTH * BATCH_NUM:
        np.concatenate((data, data), axis=-1)
        length *= 2
    data = np.stack([data[:, :, i * INPUT_LENGTH:(i + 1) * INPUT_LENGTH] for i in range(BATCH_NUM)], axis=0)
    data = np.stack([data[:, :, :, i * INPUT_LENGTH // 4:(i + 1) * INPUT_LENGTH // 4] for i in range(4)], axis=2)
    return data
