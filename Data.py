import os
import pathlib
import random
from tqdm import tqdm
import pretty_midi as pm
import numpy as np
import torch
import warnings
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data

CHANNEL_NUM = 6
CLASS_NUM = 128
INPUT_LENGTH = 2048

def piano_roll(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
    classes = [0, 3, 5, 7, 8, 9]
    piano_rolls = [_.get_piano_roll() for _ in song.instruments]
    length = np.amin([roll.shape[1] for roll in piano_rolls])
    data = np.zeros(shape=(CHANNEL_NUM, CLASS_NUM, length))
    for roll, instrument in zip(piano_rolls, song.instruments):
        if not instrument.is_drum and instrument.program // 8 in classes:
            i = classes.index(instrument.program // 8)
            data[i] = np.add(data[i], roll[:, :length])
    data_all = data
    num = random.randint(0, length - INPUT_LENGTH)
    data = data_all[:, :, num : INPUT_LENGTH + num]
    datasum = np.sum(data)
    sumall = np.sum(data_all)
    datasum = 0
    while datasum == 0 and sumall > 0:
        num = random.randint(0, length - INPUT_LENGTH)
        data = data_all[:, :, num : INPUT_LENGTH + num]
        datasum = np.sum(data)
    for datum in data:
        datum[0] += 1 - datum.sum(axis=0)
    data = data > 0
    data = np.concatenate(data, axis=0)
    return data.astype(np.float32)

def clean(x):
    x = x[0] > 0.5
    for i in range(6):
        x[i * 128, :] = 0
    x = np.split(x, 128)
    return x

def save_roll(x, step):
    x = np.concatenate(x, axis=0)
    fig = plt.figure(figsize=(72, 24))
    librosa.display.specshow(x, x_axis='time', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
    plt.title('{}'.format(step))
    fig.savefig('Samples/{}.png'.format(step))
    plt.close(fig)

def piano_rolls_to_midi(x, fs=96):
    notes = x[0].shape[0]
    midi = pm.PrettyMIDI()
    instruments = [0, 24, 40, 56, 64, 72]
    for roll, instrument in zip(x, instruments):
        current_inst = pm.Instrument(instrument)
        current_roll = np.pad(roll, [(0, 0), (1, 1)], 'constant')
        velocity_changes = np.nonzero(np.diff(current_roll).T)
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)
        for time, note in zip(*velocity_changes):
            velocity = current_roll[note, time + 1]
            time /= fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pm.Note(
                    velocity=prev_velocities[note], 
                    pitch=note, 
                    start=note_on_time[note], 
                    end=time
                )
                current_inst.notes.append(pm_note)
                prev_velocities[note] = 0
        midi.instruments.append(current_inst)
    return midi




class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    
    def __getitem__(self, index):
        data = piano_roll(self.pathlist[index])
        return data

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=32):
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
