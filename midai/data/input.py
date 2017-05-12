import numpy as np
from midai.utils import clamp, map_range
from midai.data.utils import parse_midi, filter_monophonic
from multiprocessing import Pool as ThreadPool

#TODO support model_class param w/ vals 'time-sequence' and 'event'
def from_midi(midi_paths=None, 
              raw_midi=None, 
              note_representation='absolute',
              encoding='one-hot',
              window_size=15,
              batch_size=32,
              val_split=0.20,
              shuffle=False,
              num_threads=1):
    pass

def from_midi_generator(midi_paths=None, 
                        raw_midi=None, 
                        note_representation='absolute',
                        encoding='one-hot',
                        window_size=15,
                        batch_size=32,
                        val_split=0.20,
                        shuffle=False,
                        num_threads=1):

    val_split_index = int(float(len(midi_paths)) * val_split)
    train_paths = midi_paths[0:val_split_index]
    val_paths = midi_paths[val_split_index:]

    train_gen = _get_data_generator(midi_paths, raw_midi, note_representation,
                                    encoding, window_size, batch_size, 
                                    val_split, shuffle, num_threads)

    val_gen   = _get_data_generator(midi_paths,  raw_midi, note_representation,
                                    encoding, window_size, batch_size,
                                    val_split, shuffle, num_threads)
    return train_gen, val_gen
    
def _get_data_generator(midi_paths,
                        raw_midi,
                        note_representation,
                        encoding, 
                        window_size, 
                        batch_size, 
                        val_split, 
                        shuffle, 
                        num_threads):
    if num_threads > 1:
        pool = ThreadPool(num_threads)

    load_index = 0
    max_files_in_ram = 10

    while True:
        load_files = midi_paths[load_index:load_index + max_files_in_ram]
        load_index = (load_index + max_files_in_ram) % len(midi_paths)

        # print('loading large batch: {}'.format(max_files_in_ram))
        # print('Parsing midi files...')
        # start_time = time.time()
        if num_threads > 1:
            parsed = pool.map(parse_midi, load_files)
        else:
            parsed = list(map(parse_midi, load_files))
        # print('Finished in {:.2f} seconds'.format(time.time() - start_time))
        # print('parsed, now extracting data')
        data = _windows_from_monophonic_instruments(parsed, window_size, 
                                                    note_representation, encoding)
        
        # if shuffle:
        #     # shuffle in unison
        #     tmp = list(zip(data[0], data[1]))
        #     random.shuffle(tmp)
        #     tmp = zip(*tmp)
        #     data[0] = np.asarray(tmp[0])
        #     # THIS ERRORS SOMETIMES w/:
        #     # IndexError: list index out of range
        #     data[1] = np.asarray(tmp[1])

        batch_index = 0
        while batch_index + batch_size < len(data[0]):
            # print('yielding small batch: {}'.format(batch_size))
            
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size
        
        # probably unneeded but why not
        del parsed # free the mem
        del data # free the mem
    
# returns X, y data windows from all monophonic instrument
# tracks in a pretty midi file
def _windows_from_monophonic_instruments(midi, window_size, note_representation, encoding):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = filter_monophonic(m.instruments, 1.0)
            for instrument in melody_instruments:
                if len(instrument.notes) > window_size:
                    windows = _encode_windows(instrument, 
                                              window_size,
                                              note_representation, 
                                              encoding)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return [np.asarray(X), np.asarray(y)]

def _encode_windows(pm_instrument, window_size, note_representation, encoding):
    
    if note_representation == 'absolute':
        if encoding == 'one-hot':
            return _encode_sliding_window_absolute_one_hot(pm_instrument, window_size)
    if note_representation == 'relative':
        if encoding == 'one-hot':
            return _encode_window_relative_one_hot(pm_instrument, window_size)

    raise Exception('Unsupported note_representation, encoding combo: {}, {}'
                    .format(note_representation, encoding))

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# expects pm_instrument to be monophonic.
def _encode_window_absolute_one_hot(pm_instrument, window_size):

    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # transform note velocities into 1s
    roll = (roll > 0).astype(float)
    
    # calculate the percentage of the events that are rests
    # s = np.sum(roll, axis=1)
    # num_silence = len(np.where(s == 0)[0])
    # print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows

def _encode_window_relative_one_hot(pm_instrument, window_size):
    
    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # transform note velocities into 1s
    roll = (roll > 0).astype(float)

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    roll = np.argmax(roll, axis=1)

    obj = {
        'last_played_note': 0
    } 

    def to_interval(this, last, obj):

        rest_token = 1000
        val = None

        # if this is a rest
        if this == 0:
            val = rest_token
        else:
            # if the last token was a rest
            if last == 0: 
                if obj['last_played_note'] == 0:
                    val = 0
                else:
                    val = this - obj['last_played_note']
            else:
                val = this - last

         # if the last token wasn't a rest
        if this != 0:
                # save this value for the next note on
            obj['last_played_note'] = this

        return val

    def to_one_hot(val, rest_token=1000):
        vec = np.zeros(101)
        if val == rest_token:
            vec[0] = 1
        else:    
            # to int might be creating a bug here
            index = int(map_range((-50, 50), (1, 100), clamp(val, -50, 50)))
            vec[index] = 1
        return vec

    windows = []
    for i in range(1, roll.shape[0] - window_size - 1):
        
        window_  = roll[i:i + window_size]
        predict_ = roll[i + window_size + 1]

        window = []
        for i, _ in enumerate(window_):
            window.append(to_one_hot(to_interval(window_[i], window_[i - 1], obj)))
        predict = to_one_hot(to_interval(predict_, window_[-1], obj))
        windows.append((window, predict))

    return windows