from midai.models import Model
import glob
from midai.utils import log
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import model_from_json

class KerasModel(Model):

	def init(self):
		self.name = "KerasModel"

	def load(self, experiment_dir):

		if self.ready:
			log('self.ready == True, do you really mean to load?', 'WARNING')

		if len(self.models) > 0:
			log('models list is not empty.' \
				'Have you already loaded this model?', 'WARNING')
		
		self.models = []
		num_models = len(glob.glob(os.path.join(experiment_dir, 'model_*.json')))
		for i in range(num_models):
		    with open(os.path.join(experiment_dir, 'model_{}.json'.format(i)), 'r') as f:
		        model = model_from_json(f.read())
		        log('loaded model {} from JSON'.format(i), 'verbose')

		    epoch = 0
		    path = [experiment_dir, 'checkpoints', 'model-{}_*.hdf5'.format(i)]
		    newest_checkpoint = max(glob.iglob(*path), key=os.path.getctime)

		    if newest_checkpoint: 
		       epoch = int(newest_checkpoint[-22:-19])
		       model.load_weights(newest_checkpoint)
		       log('loaded model {} weights from checkpoint {}'
		       	   .format(i, newest_checkpoint), 'verbose')

		    self.models.append(model)

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

			self.models[i].compile(loss='categorical_crossentropy', 
								   optimizer=optimizer,
								   metrics=['accuracy'])

	def _validate_learning_process(self, lp):
		
		def type_check(key, _type):
			if key in lp and type(lp[key]) != _type:
				raise Exception('type mismatch in {} in learning process. '\
					            'Expected {} got {}'.format(key, _type, type(lpy[key])))

		type_check('learning_rate', float)
		type_check('optimizer', str)
		type_check('grad_clipvalue', float)
		type_check('grad_clipnorm', float)