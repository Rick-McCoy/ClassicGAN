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

INPUT_LENGTH = 4096
pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
trainlist = pathlist[:-108]
testlist = pathlist[-108:]

def piano_roll(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
    classes = [0, 3, 5, 7, 8, 9]
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    piano_rolls = [(_.get_piano_roll(), _.program) for _ in song.instruments if not _.is_drum and _.program // 8 in classes]
    length = np.amin([roll.shape[1] for roll, _ in piano_rolls])
    data = [np.zeros(shape=(limits[i][1]-limits[i][0], length)) for i in range(6)]
    for roll, instrument in piano_rolls:
        i = classes.index(instrument // 8)
        data[i] = np.add(data[i], roll[limits[i][0]:limits[i][1], :length])
    data_all = data
    num = random.randint(0, length - INPUT_LENGTH)
    data = [data_all[i][:, num : INPUT_LENGTH + num] for i in range(6)]
    datasum = sum([np.sum(datum) for datum in data])
    sumall = sum([np.sum(datum) for datum in data_all])
    datasum = 0
    while datasum == 0 and sumall > 0:
        num = random.randint(0, length - INPUT_LENGTH)
        data = [data_all[i][:, num : INPUT_LENGTH + num] for i in range(6)]
        datasum = sum([np.sum(datum) for datum in data])
    data = np.concatenate(data, axis=0)
    data[0] += 1 - data.sum(axis = 0)
    data = data > 0
    return data.astype(np.float32)

def clean(x):
    x = x[0] > 0.5
    x[0] = 0
    return x

def save_roll(x, step):
    fig = plt.figure(figsize=(72, 24))
    librosa.display.specshow(x, x_axis='time', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
    plt.title('{}'.format(step))
    fig.savefig('Samples/{}.png'.format(step))
    plt.close(fig)

def piano_rolls_to_midi(x, fs=96):
    channels = [72, 48, 72, 48, 48, 36]
    for i in range(1, 6):
        channels[i] += channels[i - 1]
    x = np.split(x, channels)
    midi = pm.PrettyMIDI()
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    instruments = [0, 24, 40, 56, 64, 72]
    for roll, instrument, limit in zip(x, instruments, limits):
        current_inst = pm.Instrument(instrument)
        current_roll = np.pad(roll, [(limit[0], 128 - limit[1]), (1, 1)], 'constant')
        notes = current_roll.shape[0]
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
    def __init__(self, train):
        super(Dataset, self).__init__()
        if train:
            self.pathlist = trainlist
        else:
            self.pathlist = testlist
    
    def __getitem__(self, index):
        data = piano_roll(self.pathlist[index])
        return data

    def __len__(self):
        return len(self.pathlist)

class DataLoader(data.DataLoader):
    def __init__(self, batch_size, shuffle=True, num_workers=32, train=True):
        super(DataLoader, self).__init__(Dataset(train), batch_size, shuffle, num_workers=num_workers)

def Test():
    pathlist = list(pathlib.Path('Classics').glob('**/*.mid'))
    random.shuffle(pathlist)
    print(len(pathlist))
    instruments = [0, 3, 5, 7, 8, 9]
    limits = [[24, 96], [36, 84], [24, 96], [36, 84], [36, 84], [60, 96]]
    for path in tqdm(pathlist):
        song = pm.PrettyMIDI(midi_file=str(path), resolution=96)
        rolls = [(_.get_piano_roll(), _.program) for _ in song.instruments if _.program // 8 in instruments and not _.is_drum]
        length = min([_.shape[1] for _, _1 in rolls])
        data = [np.zeros(shape=(limits[i][1]-limits[i][0], length)) for i in range(6)]
        for roll, instrument in rolls:
            i = instruments.index(instrument // 8)
            data[i] = np.add(data[i], roll[limits[i][0]:limits[i][1], :length])
        if sum([np.sum(datum) for datum in data]) == 0:
            tqdm.write(str(path))

if __name__ == '__main__':
    Test()
