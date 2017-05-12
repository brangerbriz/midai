import pretty_midi
import numpy as np
from midai.utils import clamp, map_range

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def to_midi(output, 
            note_representation, 
            instrument_name='Acoustic Grand Piano',
            start_note=60):
    midis = []
    for out in output:
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=instrument_program)
        
        if note_representation == 'absolute':
            instrument.notes = _get_notes_absolute(out)
        elif note_representation == 'relative':
            instrument.notes = _get_notes_relative(out, start_note)
        else:
            raise Exception('{} is not a valid note_representation'\
                            .format(note_representation))

        # Add the cello instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
        midis.append(midi)
    return midis

def _get_notes_absolute(output, allow_represses=False):

    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0
    notes = []
    # Iterate over note names, which will be converted to note number later
    for step in output:

        note_num = np.argmax(step) - 1
        
        # a note has changed
        if allow_represses or note_num != cur_note:
            
            # if a note has been played before and it wasn't a rest
            if cur_note is not None and cur_note >= 0:            
                # add the last note, now that we have its end time
                note = pretty_midi.Note(velocity=127, 
                                        pitch=int(cur_note), 
                                        start=cur_note_start, 
                                        end=clock)
                notes.append(note)

            # update the current note
            cur_note = note_num
            cur_note_start = clock

        # update the clock
        clock = clock + 1.0 / 4
    return notes

# create a pretty midi file with a single instrument using the one-hot encoding
# output of keras model.predict.
def _get_notes_relative(output, start_note, allow_represses=False):

    def from_one_hot(vec, rest_token=1000):
        index = np.argmax(vec)
        if index == 0:
            return rest_token
        else:
            # weirdly this has to be mapped from 0-99 if to_one_hot is mapping
            # from 1-100
            return map_range((0, 99), (-50, 50), index)
    
    cur_note = None # an invalid note to start with
    cur_note_start = None
    clock = 0
    notes = []

    last_played_note = start_note

    # Iterate over note names, which will be converted to note number later
    for step in output:

        interval = from_one_hot(step)
        if interval == 1000:
            note_num = -1
        else:
            last_played_note = clamp(last_played_note + interval, 0, 127)
            note_num = last_played_note
        
        # a note has changed
        if allow_represses or note_num != cur_note:
            
            # if a note has been played before and it wasn't a rest
            if cur_note is not None and cur_note >= 0:            
                # add the last note, now that we have its end time

                note = pretty_midi.Note(velocity=127, 
                                        pitch=int(cur_note), 
                                        start=cur_note_start, 
                                        end=clock)
                notes.append(note)

            # update the current note
            cur_note = note_num
            cur_note_start = clock

        # update the clock
        clock = clock + 1.0 / 4

    return notes
