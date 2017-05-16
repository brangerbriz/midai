import os
import argparse

def from_argv(args):
	pass

def defaults():

	args = dict()

	# MISC ---------------------------------------------------------------------
	args['midai_root']  = os.path.join(os.path.dirname(__file__), '..')
	args['mode']        = 'develop' # production
	args['num_threads'] = 8

	# MODEL --------------------------------------------------------------------
	args['model']               = 'TimeSeqModel'
	args['tasks']               = ['train', 'generate'] # ['train', 'generate']
	args['load_search_path']    = os.path.join(args['midai_root'], 'trained_models', args['mode'], args['model'])
	args['load']                = None # None, 'best', 'recent', path
	args['data_dir']            = os.path.join(args['midai_root'], 'data', 'collections', '2') 

	# TRAIN --------------------------------------------------------------------
	args['data_encoding']       = 'glove-embedding' # glove-embedding one-hot
	args['model_class']         = 'sequence' # event
	args['note_representation'] = 'absolute' # relative
	args['window_size']         = 20
	args['batch_size']          = 32
	args['num_epochs']          = 10
	args['use_generator']       = True
	args['val_split']           = 0.2
	args['glove_dimension']     = 25

	args['architecture'] = [{
		"window_size": 20,
	    "input_size": 25, # 129, 101, 25, 10
		"layers": [64, 129],
		"units": ["LSTM", "Dense"],
		"activations": ["relu", "softmax"],
		"dropout": [0.5, 0.0],
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

	args['learning_process'] = [{
		'optimizer': 'nadam'
	}]

	# GENERATE -----------------------------------------------------------------
	args['num_files_to_generate'] = 10
	args['generated_file_length'] = 500
	args['seed']                  = os.path.join(args['midai_root'], 'data', 'seeds', '001.mid') 

	return args

def validate(args):
	pass