import os
from midai.utils import log

class Model:

	def __init__(self, experiment_dir=None):
		self.name = "Model"
		self.models = []
		self.experiment_dir = experiment_dir
		self.ready = False
		self.init()

	def init(self):
		pass

	# creates an experiment directory structure and returns the name
	# of the created directory
	def create_experiment_dir(self,
							  note_representation,
		                      encoding,
		                      experiment_dir=None, 
		                      midai_root=None, 
		                      mode='develop'):
		log('creating experiment directory', 'VERBOSE')
		if not experiment_dir:

			if not midai_root:
				raise Exception('midai_root not set and experiment_dir'\
				                ' not provided as an argument')

			path = [midai_root, 'trained_models', mode, self.name,
			        '{}_{}'.format(note_representation, encoding)]
			parent_dir = os.path.join(*path)

			if not os.path.exists(parent_dir):
				log('{} does not already exist, creating.'.format(parent_dir), 
					'notice')
				os.makedirs(parent_dir)

			experiments = os.listdir(parent_dir)
			experiments = [dir_ for dir_ in experiments \
						   if os.path.isdir(os.path.join(parent_dir, dir_))]

			log('{} existing directories found in {}'\
				.format(len(experiments), parent_dir), 'VERBOSE')

			most_recent_exp = 0
			for dir_ in experiments:
				try:
					most_recent_exp = max(int(dir_), most_recent_exp)
				except ValueError as e:
					# ignrore non-numeric folders in experiments/
					pass

			experiment_dir = os.path.join(parent_dir, 
				                          str(most_recent_exp + 1).rjust(3, '0')) 

		os.makedirs(experiment_dir)
		log('created {}'.format(experiment_dir), 'VERBOSE')
		
		os.makedirs(os.path.join(experiment_dir, 'checkpoints'))
		log('created {}'\
			.format(os.path.join(experiment_dir, 'checkpoints')), 'VERBOSE')
		
		os.makedirs(os.path.join(experiment_dir, 'generated'))
		log('created {}'\
			.format(os.path.join(experiment_dir, 'generated')), 'VERBOSE')
		
		self.experiment_dir = experiment_dir
		return experiment_dir

	def load(self):
		pass

	def save(self):
		pass

	def architecture(self, architectures):
		pass

	def train(self, train_data=None, val_data=None, train_gen=None, val_gen=None):

		if not self.ready:
			raise Exception('train called before model ready is True')

		if train_data is None and val_data is None:
			if not train_gen or not val_gen:
				raise Exception('If train_data and val_data are omitted '\
					            'then you must provide train_gen and val_gen')

		if not train_gen and not val_gen:
			if train_data is None or val_data is None:
				raise Exception('If train_gen and val_gen are omitted '\
					            'then you must provide train_data and val_data')

		if train_data is not None and val_data is not None and train_gen and val_gen:
			raise Exception('You cannot use both data and generators for training')

	def evaluate(self):
		pass

	def predict(self, kwargs):
		pass
