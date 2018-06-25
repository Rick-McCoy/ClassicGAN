from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pretty_midi as pm
import numpy as np
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import argparse
import os
from tqdm import tqdm
from Data import INPUT_LENGTH, CHANNEL_NUM

def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # Calculate time per pixel
    time_per_pixel = 60.0 / tempo / float(beat_resolution)
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = np.pad(piano_roll, ((0, 0), (1, 1)), 'constant', constant_values=0)
    # add padding for diff
    piano_roll_search = np.diff(piano_roll.astype(int), axis=1)
    # Iterate through all possible(128) pitches
    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[note_num] > 0).nonzero()
        start_time = time_per_pixel * (start_idx[0].astype(float))
        end_idx = (piano_roll_search[note_num] < 0).nonzero()
        end_time = time_per_pixel * (end_idx[0].astype(float))
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            # Create an Note object with corresponding note number, start time and end time
            note = pm.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # Add the note to the Instrument object
            instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)

def write_piano_rolls_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100, tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        return False
    # Create a PrettyMIDI object
    midi = pm.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for program_num, is_drum_i, piano_roll in zip(program_nums, is_drum, piano_rolls):
        # Create an Instrument object
        instrument = pm.Instrument(program=program_num, is_drum=is_drum_i)
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)

def unpack_sample(name='', concat=False):
    if name == '':
        pathlist = list(pathlib.Path('Samples').glob('**/*.npy'))
        name = str(pathlist[-1])
    if not os.path.exists(name):
        os.mkdir(name)
    savename = name + '/' + name.split('/')[-1]
    if not '.npy' in name:
        name = name + '.npy'
    samples = np.load(name) > 0
    program_nums = [0, 24, 40, 56, 64, 72]
    is_drum = [False] * CHANNEL_NUM
    if concat:
        sample = np.concatenate([i for i in samples], axis=-1)
        write_piano_rolls_to_midi(sample, program_nums=program_nums, is_drum=is_drum, filename=savename + '.mid')
        tqdm.write(name + '.mid')
        for i, piano_roll in enumerate(sample):
            fig = plt.figure(figsize=(12, 4))
            librosa.display.specshow(piano_roll, x_axis='time', y_axis='cqt_note', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
            plt.title(savename + '_' + pm.program_to_instrument_name(program_nums[i]))
            fig.savefig(savename + '_' + str(i) + '.png')
            plt.close(fig)
        return
    for id, sample in enumerate(samples):
        write_piano_rolls_to_midi(sample, program_nums=program_nums, is_drum=is_drum, filename=savename + '_' + str(id) + '.mid')
        tqdm.write(savename + '_' + str(id) + '.mid')
        for i, piano_roll in enumerate(sample):
            fig = plt.figure(figsize=(12, 4))
            librosa.display.specshow(piano_roll, x_axis='time', y_axis='cqt_note', hop_length=1, sr=96, fmin=pm.note_number_to_hz(12))
            plt.title(savename + '_' + pm.program_to_instrument_name(program_nums[i]))
            fig.savefig(savename + '_' + str(id) + '_' + str(i) + '.png')
            plt.close(fig)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', type=str, default='', help='Determines which npy file to sample. Defaults to last file.')
    parser.add_argument('-c', '--concat', type=bool, default=False, help='Enable Concatenation. Defaults to False.')
    args = parser.parse_args()
    unpack_sample(name=args.sample, concat=args.concat)
if __name__ == '__main__':
    main()
