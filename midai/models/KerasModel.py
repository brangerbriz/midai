from midai.models import Model
import glob
from midai.utils import log
from keras.models import model_from_json

class KerasModel(Model):

	def init(self):
		self.name = "KerasModel"

	def compile(self):
		pass

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
