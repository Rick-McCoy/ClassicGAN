from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pretty_midi as pm
import numpy as np

def get_instrument(piano_roll, program_num, is_drum, velocity=100, tempo=120.0, beat_resolution=24):
    tpp = 60.0 / tempo / float(beat_resolution)
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 84), dtype=int), piano_roll, np.zeros((1, 84), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    instrument = pm.Instrument(program=program_num, is_drum=is_drum)
    for note_num in range(12, 96):
        start_idxs = (piano_roll_search[:, note_num] > 0).nonzero()
        start_times = tpp * (start_idxs[0].astype(float))
        end_idxs = (piano_roll_search[:, note_num] < 0).nonzero()
        end_times = tpp * (end_idxs[0].astype(float))
        for idx, start_time in enumerate(start_times):
            note = pm.Note(velocity=velocity, pitch=note_num, start=start_times[idx], end=end_times[idx])
            instrument.notes.append(note)
    instrument.notes.sort(key=lambda note: note.start)
    return instrument

def get_midi(piano_rolls, program_nums=None, is_drum=None, velocity=100, tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums):
        raise ValueError("piano_rolls and program_nums should have the same length")
    if len(piano_rolls) != len(is_drum):
        raise ValueError("piano_rolls and is_drum should have the same length")
    if program_nums is None:
        program_nums = [0] * len(piano_rolls)
    if is_drum is None:
        is_drum = [False] * len(piano_rolls)
    song = pm.PrettyMIDI(initial_tempo=tempo)
    for idx, piano_roll in enumerate(piano_rolls):
        instrument = get_instrument(piano_roll, program_nums[idx], is_drum[idx], velocity, tempo, beat_resolution)
        song.instruments.append(instrument)
    return song

def main():
    print('Hello')
    #pathlist = list(pathlib.Path('Samples').glob('**/*.npy'))
    #program_nums = [i * 8 for i in range(17)]
    #program_nums[16] = 0
    #is_drum = [False for i in range(17)]
    #is_drum[16] = True
    #index = '002000'
    #pathlist = ['Samples/song_' + index + '/song_' + index + '.npy']
    #for num, path in enumerate(pathlist):
    #    song = pm.PrettyMIDI(path)
    #    for j, ins in enumerate(song.instruments):
    #        print(ins)
    #        fig = plt.figure(figsize=(12, 4))
    #        librosa.display.specshow(ins.get_piano_roll(), hop_length=1, sr=100, x_axis='time', y_axis='cqt_note', fmin=pm.note_number_to_hz(12))
    #        plt.title('midi piano roll: ' + 'epoch_05000.npy' + ' ' + (pm.program_to_instrument_class(ins.program) if not ins.is_drum else 'drum'))
    #        fig.savefig(str(path) + '_' + str(j) + '.png')
    #        plt.close(fig)
    #    data = np.transpose(np.squeeze(np.load(path)), (1, 0, 2, 3))
    #    data = (data + 1) * 100
    #    for j in range(3):
    #        for i, piano in enumerate(data[j]):
    #            fig = plt.figure(figsize=(12, 4))
    #            librosa.display.specshow(piano, hop_length=1, sr=100, x_axis='time', y_axis='cqt_note', fmin=pm.note_number_to_hz(12))
    #            plt.title('midi piano roll: ' + 'song_' + index + '.npy' + ' ' + (pm.program_to_instrument_name(i * 8) if i < 16 else 'drums'))
    #            fig.savefig(str(path) + '_' + str(j) + '_' + str(i) + '.png')
    #            plt.close(fig)
    #        print(piano.shape)
    #        for j, ins in enumerate(piano):
    #            song = get_midi(piano, program_nums, is_drum)
    #            print(path + str(i) + '.mid')
    #            song.write(path + str(i) + '.mid')
if __name__ == '__main__':
    main()
