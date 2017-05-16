import os, pudb
import midai
from midai.models.base import KerasRNNModel
from midai.utils import log
from midai.data.utils import save_midi
from midai.arguments import defaults

def get_model(args):
	
	model = KerasRNNModel()
	model.name = "TimeSeqModel"

	# load an existing model
	if 'load' in args and args['load']:
		if args['load'] == 'best' or args['load'] == 'recent':
			kwargs = {
				'best': args['load'] == 'best',
				'recent': args['load'] == 'recent'
			}
			model.load(args['load_search_path'], **kwargs)
		else:
			model.load(args['load'], best=True, recent=False)
	else: # create a new model
		model.create_experiment_dir(args['note_representation'],
									args['data_encoding'],
		                            midai_root=args['midai_root'])

	if not model.ready:
		model.architecture(args['architecture'])
		model.save(model.experiment_dir)

	model.compile(args['learning_process'])
	return model

def get_data(args):

	midi_paths = midai.data.utils.get_midi_paths(args['data_dir'])

	kwargs = {
		"midi_paths": midi_paths,
        "note_representation": args['note_representation'],
        "encoding": args['data_encoding'],
        "window_size": args['window_size'],
        "val_split": args['val_split'],
        "glove_dimension": args['glove_dimension'],
        "num_threads": args['num_threads']
	}

	if args['use_generator']:
		_train, _val = midai.data.input.from_midi_generator(**kwargs)
	else:
		_train, _val = midai.data.input.from_midi(**kwargs)
	return (_train, _val), midi_paths


def train(args, model, data, num_midi_files):

	kwargs = dict()
	kwargs['num_midi_files'] = num_midi_files
	kwargs['num_epochs']     = args['num_epochs']
	kwargs['batch_size']     = args['batch_size']

	if args['use_generator']:
		kwargs['train_gen']  = data[0]
		kwargs['val_gen']    = data[1]
	else:
		kwargs['train_data'] = data[0]
		kwargs['val_data']   = data[1]
	model.train(**kwargs)

def generate(args, model, data):
	if args['seed']:
		_, val = midai.data.input.from_midi(midi_paths=[args['seed']],
		                                    note_representation=args['note_representation'],
	                                        encoding=args['data_encoding'],
	                                        window_size=args['window_size'],
	                                        val_split=args['val_split'],
	                                        glove_dimension=args['glove_dimension'])
		X = val[0] # use only the first window
	else:
		X, _ = next(data[1])

	output = model.generate(X, args['window_size'], 
		                    args['generated_file_length'], 
		                    args['num_files_to_generate'],
		                    args['data_encoding'])
	midis = midai.data.output.to_midi(output[0], args['note_representation'])
	save_midi(midis, os.path.join(model.experiment_dir, 'generated'))

def run(args):
	
	model        = get_model(args)
	data, paths  = get_data(args)
	
	if 'train' in args['tasks']:
		train(args, model, data, len(paths))
	if 'generate' in args['tasks']:
		generate(args, model, data)

if __name__ == '__main__':
	run(defaults())