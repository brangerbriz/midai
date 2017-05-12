import os
import midai
from midai.models.base import KerasRNNModel
from midai.utils import log
from midai.data.utils import save_midi
from midai.arguments import defaults

def get_model(args):
	
	model = KerasRNNModel()
	model.name = "TimeSeqModel"

	# load an existing model
	if 'load' in args:
		if args['load'] == 'best' or args['load'] == 'recent':
			paths = [args['midai_root'], 'trained_models', args['mode'], model.name]
			kwargs = {
				'best': args['load'] == 'best',
				'recent': args['load'] == 'recent'
			}
			model.load(os.path.join(*paths), **kwargs)
		else:
			model.load(args['load'], best=True, recent=False)
	else: # create a new model
		model.create_experiment_dir()

	if not model.ready:
		model.architecture(args['architecture'])
		model.save(model.experiment_dir)

	model.compile(args['learning_process'])
	return model

def get_data(args):

	midi_paths = midai.data.utils.get_midi_paths(args['data_dir'])

	train_gen, val_gen = midai.data.input.from_midi_generator(midi_paths=midi_paths,
	                                                          note_representation=args['note_representation'],
	                                                          encoding=args['data_encoding'],
	                                                          window_size=args['window_size'],
	                                                          val_split=args['val_split'])
	return (train_gen, val_gen), midi_paths


def train(args, model, data, num_midi_files):
	model.train(num_midi_files=num_midi_files, train_gen=data[0], val_gen=data[1])

def generate(args, model, data):
	X, _ = next(data[1])
	output = model.generate(X, args['window_size'], 
		                    args['generated_file_length'], 
		                    args['num_files_to_generate'])
	midis = midai.data.output.to_midi(output[0], args['note_representation'])
	save_midi(midis, os.path.join(model.experiment_dir, 'generated'))

def run(args):
	
	args['tasks'] = ['generate']

	model        = get_model(args)
	data, paths  = get_data(args)
	
	if 'train' in args['tasks']:
		train(args, model, data, len(paths))
	if 'generate' in args['tasks']:
		generate(args, model, data)

if __name__ == '__main__':
	run(defaults())