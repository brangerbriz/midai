import os, pudb
import numpy as np
from midai.utils import clamp, map_range, log
from midai.data.utils import parse_midi, filter_monophonic, split_data
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
              num_threads=1,
              glove_dimension=10):
    if num_threads > 1:
        pool = ThreadPool(num_threads)
        parsed = pool.map(parse_midi, midi_paths)
    else:
        parsed = list(map(parse_midi, midi_paths))
    
    data = _windows_from_monophonic_instruments(parsed, window_size, 
                                                note_representation, encoding, 
                                                glove_dimension)

    # convert data from (X0-n, y0-n) to ((X0, y0), (X1, y1), ...) format
    data = list(zip(data[0], data[1]))

    if shuffle:
        data = np.random.permutation(data)

    train, val = split_data(data, val_split)
    return np.asarray(list(zip(*train))).tolist(), np.asarray(list(zip(*val))).tolist()

def from_midi_generator(midi_paths=None, 
                        raw_midi=None, 
                        note_representation='absolute',
                        encoding='one-hot',
                        window_size=15,
                        batch_size=32,
                        val_split=0.20,
                        shuffle=False,
                        num_threads=1,
                        glove_dimension=10):

    train_paths, val_paths = split_data(midi_paths, val_split)
    pudb.set_trace()

    train_gen = _get_data_generator(train_paths, note_representation,
                                    encoding, window_size, batch_size, 
                                    shuffle, num_threads, glove_dimension)

    val_gen   = _get_data_generator(val_paths, note_representation,
                                    encoding, window_size, batch_size,
                                    shuffle, num_threads, glove_dimension)
    return train_gen, val_gen

def one_hot_2_glove_embedding(X):
    
    if not _glove_embeddings:
        raise Exception('glove embeddings have not been loaded. '\
                        'Load with load_glove_embeddings(...)')

    # store glove_embedding Xs in a temp buff
    buf = []
    for j, x in enumerate(X):
       # the one-hot encoding stores rests as the first element, but our
       # embedding table stores it as the 128th element, so we pop the first
       # value off of the front and append it to the back.
       rest = x[0] 
       x = np.delete(x, 0)
       x = np.append(x, rest)
       index = np.argmax(x)
       buf.append(_glove_embeddings[index])
    return buf

def load_glove_embeddings(dim, glove_path):
    # skip if done
    if _glove_embeddings:
        return

    global _glove_embeddings
    _glove_embeddings = []

    # parse glove embeddings csv transforming TRACK_NUM to 128 and <unk> to 129
    with open(os.path.join(glove_path, 'vectors_d{}.txt'.format(dim)), 'r') as f:
        for line in f.readlines():
            split = line.split()
            if split[0] == '<unk>': split[0] = 130
            elif split[0] == 'TRACK_START': split[0] = 129
            _glove_embeddings.append(np.asarray([float(x) for x in split]))

    # add random rest vector as the 128th row
    with open(os.path.join(glove_path, 'rest.txt')) as f:
        vec = [128] + [float(x) for x in f.read().split(' ')][0:dim]
        _glove_embeddings.append(vec)

    # sort the list by index (numeric key)
    _glove_embeddings.sort(key=lambda x: x[0])

    # remove index
    for i, _ in enumerate(_glove_embeddings):
        _glove_embeddings[i] = np.delete(_glove_embeddings[i], 0)

    log('loaded GloVe vector embeddings with dimension: {}'.format(dim), 'VERBOSE')
_glove_embeddings = None
    
def _get_data_generator(midi_paths,
                        note_representation,
                        encoding, 
                        window_size, 
                        batch_size, 
                        shuffle, 
                        num_threads,
                        glove_dimension):
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
                                                    note_representation, encoding, 
                                                    glove_dimension)
        
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
def _windows_from_monophonic_instruments(midi, 
                                         window_size, 
                                         note_representation, 
                                         encoding, 
                                         glove_dimension):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = filter_monophonic(m.instruments, 1.0)
            for instrument in melody_instruments:
                # WARNING: This is an event model style check but it is also
                # currently being applied to the time sequence model.
                if len(instrument.notes) > window_size:
                    windows = _encode_windows(instrument, 
                                              window_size,
                                              note_representation, 
                                              encoding, glove_dimension)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
                else:
                    # log('Fewer notes than window_size permits, skipping instrument', 'WARNING')
                    pass
    return [np.asarray(X), np.asarray(y)]

def _encode_windows(pm_instrument, window_size, note_representation, encoding, glove_dimension):
    
    if encoding == 'glove-embedding' and not _glove_embeddings:
        load_glove_embeddings(glove_dimension, '/home/bbpwn2/Documents/code/midai/data/embeddings/glove')

    if note_representation == 'absolute':
        if encoding == 'one-hot':
            return _encode_window_absolute_one_hot(pm_instrument, window_size)
        elif encoding == 'glove-embedding':
            return _encode_window_absolute_glove_embedding(pm_instrument, window_size)
    if note_representation == 'relative':
        if encoding == 'one-hot':
            return _encode_window_relative_one_hot(pm_instrument, window_size)
        elif encoding == 'glove-embedding':
            pass

    raise Exception('Unsupported note_representation, encoding combo: {}, {}'
                    .format(note_representation, encoding))

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# expects pm_instrument to be monophonic.
def _encode_window_absolute_one_hot(pm_instrument, window_size):

    roll = np.copy(pm_instrument.get_piano_roll(fs=16).T)

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
        windows.append([roll[i:i + window_size], roll[i + window_size + 1]])
    return windows

def _encode_window_relative_one_hot(pm_instrument, window_size):
    
    roll = np.copy(pm_instrument.get_piano_roll(fs=16).T)

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

def _encode_window_absolute_glove_embedding(pm_instrument, window_size):

    # leverage existing one-hot function
    windows = _encode_window_absolute_one_hot(pm_instrument, window_size)
    
    # for each X, y window pair
    for i, window in enumerate(windows):
        windows[i][0] = one_hot_2_glove_embedding(windows[i][0])
    return windows
                   