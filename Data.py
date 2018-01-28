from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pretty_midi as pm
import numpy as np
import pathlib
from tqdm import tqdm

channel_num = 17
class_num = 72
input_length = 512

def roll(path):
    try:
        song = pm.PrettyMIDI(str(path))
    except:
        return FileNotFoundError
    if len(song.instruments) < 2:
        return FileNotFoundError
    length = np.min([i.get_piano_roll().shape[1] for i in song.instruments])
    t = np.max([i.program // 8 for i in song.instruments])
    if t < 1 or length == 0:
        return FileNotFoundError
    length = length if length < input_length * 50 else input_length * 50
    data = np.zeros(shape=(channel_num, class_num, length))
    for i in song.instruments:
        #if i.is_drum:
        #    data[16] = np.add(data[16], i.get_piano_roll()[12:96, :length])
        #else:
        #    data[i.program // 8] = np.add(data[i.program // 8],
        #    i.get_piano_roll()[12:96, :length])
        if not i.is_drum and i.program < (data.shape[0] + 1) * 8:
            data[i.program // 8] = np.add(data[i.program // 8], i.get_piano_roll()[24:96, :length])
    data = np.transpose(data, (1, 2, 0)) > 0
    #if np.sum(data[:, :, 1:]) == 0:
    #    return FileNotFoundError
    if np.sum(data) == 0:
        return FileNotFoundError
    data = (data - 0.5) * 2
    return length, data
