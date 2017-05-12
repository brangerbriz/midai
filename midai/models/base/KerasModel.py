import os, glob, time, random, copy
import numpy as np
from midai.models.base import Model
from midai.utils import log
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

class KerasModel(Model):

    def init(self):
        self.name = "KerasModel"

    # TODO: Add best=True, newest=False params and search
    # dir recursively
    def load(self, experiment_dir):

        if self.ready:
            log('model ready is True, do you really mean to load?', 'WARNING')

        if len(self.models) > 0:
            log('models list is not empty.' \
                'Have you already loaded this model?', 'WARNING')
        
        self.models = []
        num_models = len(glob.glob(os.path.join(experiment_dir, 'model_*.json')))
        if num_models < 1:
            message = 'No models found in {}'.format(experiment_dir)
            log(message, 'ERROR')
            raise Exception(message)

        for i in range(num_models):
            with open(os.path.join(experiment_dir, 'model_{}.json'.format(i)), 'r') as f:
                model = model_from_json(f.read())
                log('loaded model {} from JSON'.format(i), 'VERBOSE')

            epoch = 0
            path = [experiment_dir, 'checkpoints', 'model_{}*.hdf5'.format(i)]
            newest_checkpoint = max(glob.glob(os.path.join(*path)), key=os.path.getctime)

            if newest_checkpoint: 
               epoch = int(newest_checkpoint[-22:-19])
               model.load_weights(newest_checkpoint)
               log('loaded model {} weights from checkpoint {}'
                   .format(i, newest_checkpoint), 'VERBOSE')

            self.models.append(model)

        self.experiment_dir = experiment_dir
        self.ready = True

    def save(self, experiment_dir):
        for i, model in enumerate(self.models):
            with open(os.path.join(experiment_dir, 'model_{}.json'.format(i)), 'w') as f:
                log('saved model {} to {}'\
                    .format(i, os.path.join(experiment_dir, 
                                            'model_{}.json'.format(i))), 
                    'verbose')
                f.write(model.to_json())


    def compile(self, learning_processes):

        if not self.ready:
            raise Exception('compile called before model ready is True')
        print(len(self.models))
        if len(learning_processes) != len(self.models):
            raise Exception('element size mismatch between learning_processes'\
                            ' and number of models')

        for i, lp in enumerate(learning_processes):
        
            self._validate_learning_process(lp)

            kwargs = {}
            if 'grad_clipvalue' in lp and 'optimizer' in lp:
                kwargs['clipvalue'] = lp['grad_clipvalue']

            if 'grad_clipnorm' in lp and 'optimizer' in lp:
                kwargs['clipnorm'] = lp['grad_clipnorm']

            if 'learning_rate' in lp:
                kwargs['lr'] = lp['learning_rate']

            if 'optimizer' in lp:

                # select the optimizers
                if lp['optimizer'] == 'sgd':
                    optimizer = SGD(**kwargs)
                elif lp['optimizer'] == 'rmsprop':
                    optimizer = RMSprop(**kwargs)
                elif lp['optimizer'] == 'adagrad':
                    optimizer = Adagrad(**kwargs)
                elif lp['optimizer'] == 'adadelta':
                    optimizer = Adadelta(**kwargs)
                elif lp['optimizer'] == 'adam':
                    optimizer = Adam(**kwargs)
                elif lp['optimizer'] == 'adamax':
                    optimizer = Adamax(**kwargs)
                elif lp['optimizer'] == 'nadam':
                    optimizer = Nadam(**kwargs)
                else:
                    raise Exception('{} is not a supported optimizer'\
                                    .format(lp['optimizer']))
            else:
                optimizer = Adam()

            if 'loss' in lp:
                loss = lp['loss']
            else:
                loss = 'categorical_crossentropy'

            if 'metrics' in lp:
                metrics = lp['metrics']
            else:
                metrics = ['accuracy']

            self.models[i].compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, 
              num_midi_files,
              model_index=None,
              train_data=None, 
              val_data=None, 
              train_gen=None, 
              val_gen=None,
              num_epochs=10,
              batch_size=32):

        def _train_model(model, callbacks):
            
            start_time = time.time()

            kwargs = {
                "epochs": num_epochs,
                "callbacks": callbacks,
                "verbose": 1,
            }

            if train_gen and val_gen:
                # this is a somewhat magic number which is the average number of length-20 windows
                # calculated from ~5K MIDI files from the Lakh MIDI Dataset.
                magic_number = 827
                kwargs['generator'] = train_gen
                kwargs['validation_data'] = val_gen
                kwargs['steps_per_epoch'] = num_midi_files * magic_number / batch_size
                kwargs['validation_steps'] = num_midi_files * 0.2 * magic_number / batch_size
                history = model.fit_generator(**kwargs)
            else:
                kwargs['x'] = train_data
                kwargs['y'] = val_data
                kwargs['batch_size'] = batch_size
                kwargs['shuffle'] = False
                history = model.fit(**kwargs)

            log('Finished training model in {:.2f} seconds'.format(time.time() - start_time), 'NOTICE')
            return history

        super().train(train_data, val_data, train_gen, val_gen)
        histories = []
        if not model_index:
            for i, model in enumerate(self.models):
                histories.append(_train_model(model, self.callbacks(i)))
        else:
            if not model_index in range(len(self.models)):
                raise Exception('Invalid model_index {}'.format(model_index))
            histories.append(_train_model(self.models[model_index], self.callbacks(model_index)))

        return histories

    def generate(self, seeds, window_size, length, num_to_gen):
    
        def gen(model, seed, window_size, length):
            
            generated = []
            # ring buffer
            buf = np.copy(seed).tolist()
            while len(generated) < length:
                arr = np.expand_dims(np.asarray(buf), 0)
                pred = model.predict(arr)
                
                # argmax sampling (NOT RECOMMENDED), or...
                # index = np.argmax(pred)

                # prob distrobuition sampling
                index = np.random.choice(range(0, seed.shape[1]), p=pred[0])
                pred = np.zeros(seed.shape[1])

                pred[index] = 1
                generated.append(pred)
                buf.pop(0)
                buf.append(pred)

            return generated

        per_model = []

        for model in self.models:
            generated = []
            for i in range(0, num_to_gen):
                seed = seeds[random.randint(0, len(seeds) - 1)]
                generated.append(gen(model, seed, window_size, length))
                log('generated data of length {}'.format(length), 'VERBOSE')
            per_model.append(copy.deepcopy(generated))
        print('num files: {}'.format(len(per_model)))
        print(np.sum(per_model, axis=1))
        return per_model

    def callbacks(self, model_index):
    
        callbacks = []
        
        # save model checkpoints
        filepath = os.path.join(self.experiment_dir, 
                                'checkpoints', 
                                'model_' + str(model_index) + 
                                '-checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

        callbacks.append(ModelCheckpoint(filepath, 
                                         monitor='val_acc', 
                                         verbose=1, 
                                         save_best_only=True, 
                                         mode='max'))

        callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                           factor=0.5, 
                                           patience=3, 
                                           verbose=1, 
                                           mode='auto', 
                                           epsilon=0.0001, 
                                           cooldown=0, 
                                           min_lr=0))

        callbacks.append(TensorBoard(log_dir=os.path.join(self.experiment_dir, 'tensorboard-logs'), 
                                    histogram_freq=0, 
                                    write_graph=True, 
                                    write_images=False))

        return callbacks

    def _validate_learning_process(self, lp):
        
        def type_check(key, _type):
            if key in lp and type(lp[key]) != _type:
                raise Exception('type mismatch in {} in learning process. '\
                                'Expected {} got {}'.format(key, _type, type(lpy[key])))

        type_check('learning_rate', float)
        type_check('optimizer', str)
        type_check('grad_clipvalue', float)
        type_check('grad_clipnorm', float)