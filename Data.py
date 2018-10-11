import os
import pathlib
import random
from tqdm import tqdm
import pretty_midi as pm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data

CHANNEL_NUM = 6
CLASS_NUM = 128
INPUT_LENGTH = 4096

def piano_roll(path):
    try:
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
    except:
        tqdm.write('Error Opening')
        raise Exception
    classes = [0, 3, 5, 7, 8, 9]
    piano_rolls = [_.get_piano_roll() for _ in song.instruments]
    length = np.min([_.shape[1] for _ in piano_rolls])
    if length < INPUT_LENGTH:
        tqdm.write('Too short')
        raise Exception
    maxsum = 0
    for roll in piano_rolls:
        maxsum += np.amax(roll)
    if maxsum == 0:
        tqdm.write('No notes')
        raise Exception
    data = np.zeros(shape=(CHANNEL_NUM, CLASS_NUM, INPUT_LENGTH))
    for roll, instrument in zip(piano_rolls, song.instruments):
        if not instrument.is_drum and instrument.program // 8 in classes:
            i = classes.index(instrument.program // 8)
            data[i] = np.add(data[i], roll[:, :INPUT_LENGTH])
    data = data > 0
    data = np.concatenate(data, axis=0)
    return data.astype(np.float32)

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        self.pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    
    def __getitem__(self, index):
        path = self.pathlist[index]
        try:
            data = piano_roll(path)
        except:
            path = self.pathlist[index + 1]
            try:
                data = piano_roll(path)
            except:
                raise Exception
        return data

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, receptive_fields, batch_size, shuffle=True, num_workers=4):
        super(DataLoader, self).__init__(Dataset(), batch_size, shuffle, num_workers=num_workers)

def Test():
    pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    random.shuffle(pathlist)
    lengthlist = []
    count = 0
    for path in tqdm(pathlist[:100]):
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
        piano_rolls = [_.get_piano_roll() for _ in song.instruments]
        length = np.min([_.shape[1] for _ in piano_rolls])
        lengthlist.append(length)
        if length >= INPUT_LENGTH:
            count += 1

    plt.hist(lengthlist, 200, facecolor='green')
    plt.xlabel('Length')
    plt.ylabel('Number of Occurrence')
    plt.axis([0, 40000, 0, 15])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    Test()
