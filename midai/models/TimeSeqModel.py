import os
import midai.data as data
from midai.models.base import KerasRNNModel
from midai.utils import log

os.environ['MIDAI_ROOT'] = os.path.join(os.path.dirname(__file__), '../..')

model = KerasRNNModel()
model.name = "TimeSeqModel"

# model.create_experiment_dir()

paths = [os.environ['MIDAI_ROOT'], 'trained_models', 'develop', model.name]
model.load(os.path.join(*paths, '004'))

if not model.ready:
	architecture = [{
			"window_size": 20,
		    "input_size": 101,
			"layers": [64, 101],
			"units": ["LSTM", "LSTM", "Dense"],
			"activations": ["relu", "relu", "softmax"],
			"dropout": [0.2, 0.5, 0.0],
			# "use_bias": [True, True, True],
			# "kernel_regularizer_l1": [0.1, 0.0, 0.1],
			# "kernel_regularizer_l2": [0.1, 0.0, 0.1],
			# "bias_regularizer_l1": [0.1, 0.0, 0.1],
			# "bias_regularizer_l2": [0.1, 0.0, 0.1],
			# "recurrent_regularizer_l1": [0.1, 0.0, 0.1],
			# "recurrent_regularizer_l2": [0.1, 0.0, 0.1],
			# "activity_regularizer_l1": [0.1, 0.0, 0.1],
			# "activity_regularizer_l2": [0.1, 0.0, 0.1],
		}]


	model.architecture(architecture)
	model.save(model.experiment_dir)

learning_process = [{
	'optimizer': 'nadam'
}]

model.compile(learning_process)
print(model.models[0].summary())

data_dir = os.path.join(os.environ['MIDAI_ROOT'], 'data/collections/2')
midi_paths = data.utils.get_midi_paths(data_dir)

train_gen, val_gen = data.input.from_midi_generator(midi_paths=midi_paths,
	                                                note_representation='relative',
	                                                encoding='one-hot',
	                                                window_size=20,
	                                                val_split=0.2)

# model.train(num_midi_files=len(midi_paths), train_gen=train_gen, val_gen=val_gen)

X, _ = next(val_gen)
output = model.generate(X, 20, 1000, 10)
midis = data.output.to_midi(output[0], 'relative')
data.utils.save_midi(midis, os.path.join(model.experiment_dir, 'generated'))
