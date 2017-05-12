import os
import pretty_midi
import numpy as np
from midai.utils import log

def get_midi_paths(data_dir):
	return [os.path.join(data_dir, path) for path in os.listdir(data_dir) \
			if '.mid' in path or '.midi' in path]
	
def parse_midi(path):
    midi = None
    with open(path, 'r+b') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            midi.remove_invalid_notes()
        except Exception as e:
            log(e, 'WARNING')
            pass
    return midi

def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else: # no notes of any kind
        return 0.0
    
def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if \
            get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]

def save_midi(pm_midis, folder):
    for i, midi in enumerate(pm_midis):
        file = os.path.join(folder, '{}.mid'.format(i + 1))
        midi.write(file)
        log('saved {} to disk'.format(file), 'VERBOSE')