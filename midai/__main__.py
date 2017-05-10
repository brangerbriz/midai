import os
from midai.models import KerasRNNModel
from midai.utils import log

os.environ['MIDAI_ROOT'] = os.path.join(os.path.dirname(__file__), '..')

model = KerasRNNModel()
model.create_experiment_dir()

architecture = [{
		"window_size": 20,
	    "input_size": 129,
		"layers": [64, 64, 129],
		"activations": ["relu", "relu", "relu"],
		"dropout": [0.2, 0.5, 0.0],
		"units": ["LSTM", "LSTM", "Dense"],
		"use_bias": [True, True, True],
		"kernel_regularizer_l1": [0.1, 0.0, 0.1],
		"kernel_regularizer_l2": [0.1, 0.0, 0.1],
		"bias_regularizer_l1": [0.1, 0.0, 0.1],
		"bias_regularizer_l2": [0.1, 0.0, 0.1],
		"recurrent_regularizer_l1": [0.1, 0.0, 0.1],
		"recurrent_regularizer_l2": [0.1, 0.0, 0.1],
		"activity_regularizer_l1": [0.1, 0.0, 0.1],
		"activity_regularizer_l2": [0.1, 0.0, 0.1],
		"bias_initializer":[None, None, None], 
		"kernel_initializer":[None, None, None]
	}]

model.architecture(architecture)
