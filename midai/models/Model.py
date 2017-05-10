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
	def create_experiment_dir(self, experiment_dir=None, stage='develop'):
		log('creating experiment directory', 
			'verbose')
		if not experiment_dir:
			if not os.getenv('MIDAI_ROOT'):
				raise Exception('MIDAI_ROOT env var not set and experiment_dir'\
				                ' not provided as an argument')

			path = [os.getenv('MIDAI_ROOT'), 'trained_models', stage, self.name]
			parent_dir = os.path.join(*path)

			if not os.path.exists(parent_dir):
				log('{} does not already exist, creating.'.format(parent_dir), 
					'notice')
				os.makedirs(parent_dir)

			experiments = os.listdir(parent_dir)
			experiments = [dir_ for dir_ in experiments \
						   if os.path.isdir(os.path.join(parent_dir, dir_))]

			log('{} existing directories found in {}'\
				.format(len(experiments), parent_dir), 'verbose')

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
		log('created {}'.format(experiment_dir), 'verbose')
		
		os.makedirs(os.path.join(experiment_dir, 'checkpoints'))
		log('created {}'\
			.format(os.path.join(experiment_dir, 'checkpoints')), 'verbose')
		
		os.makedirs(os.path.join(experiment_dir, 'generated'))
		log('created {}'\
			.format(os.path.join(experiment_dir, 'generated')), 'verbose')
		
		self.experiment_dir = experiment_dir
		return experiment_dir

	def load(self):
		pass

	def save(self):
		pass

	def architecture(self, architecture):
		pass

	def train(self, kwargs):
		pass

	def evaluate(self):
		pass

	def predict(self, kwargs):
		pass
