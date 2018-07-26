from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pretty_midi as pm
import numpy as np
import pathlib
import random
import os
from tqdm import tqdm

CHANNEL_NUM = 6
CLASS_NUM = 128
INPUT_LENGTH = 512
BATCH_SIZE = 16
#0 Piano: 22584
#1 Chromatic Percussion: 216
#2 Organ: 500
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
        tqdm.write('Error while opening')
        raise Exception
    classes = [0, 3, 5, 7, 8, 9]
    piano_rolls = [i.get_piano_roll() for i in song.instruments]
    length = np.min([i.shape[1] for i in piano_rolls])
    if length < INPUT_LENGTH:
        tqdm.write('Too short')
        raise Exception
    data = np.zeros(shape=(CHANNEL_NUM, CLASS_NUM, length))
    for piano_roll, instrument in zip(piano_rolls, song.instruments):
        if not instrument.is_drum and instrument.program // 8 in classes:
            id = classes.index(instrument.program // 8)
            data[id] = np.add(data[id], piano_roll[:, :length])
    if np.amax(data) == 0:
        tqdm.write('No notes')
        raise Exception
    data = data > 0
    data = np.split(data[:, :, :length // INPUT_LENGTH * INPUT_LENGTH], indices_or_sections=length // INPUT_LENGTH, axis=-1)
    onoff = [np.array([np.amax(track) > 0 for track in datum]) for datum in data]
    return data, onoff

def build_dataset():
    pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    random.shuffle(pathlist)
    if not os.path.exists('Dataset'):
        os.mkdir('Dataset')
    writer = tf.python_io.TFRecordWriter('Dataset/cond_dataset.tfrecord')
    for path in tqdm(pathlist):
        try:
            data, onoff = roll(str(path))
        except:
            continue
        for datum, act in zip(data, onoff):
            packed_data = np.packbits(datum).tostring()
            packed_act = np.packbits(act)
            feature = {
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[packed_data])), 
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[packed_act]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            serialized = example.SerializeToString()
            writer.write(serialized)
    writer.close()

def main():
    build_dataset()

if __name__ == '__main__':
    main()
