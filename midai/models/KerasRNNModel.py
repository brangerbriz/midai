from midai.models import KerasModel 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.regularizers import l1_l2

class KerasRNNModel(KerasModel):

	# [{"window_size": 20,
	#   "input_size": 129
	# 	"layers": [64, 64, 129],
	# 	"activations": ["relu", "relu", "relu"],
	# 	"dropout": [0.2, 0.5, 0.0],
	# 	"units": ["LSTM", "LSTM", "Dense"],
	# 	"use_bias": [true, true, true],
	# 	"kernel_regularizer_l1": [0.1, 0.0, 0.1],
	# 	"kernel_regularizer_l2": [0.1, 0.0, 0.1],
	# 	"bias_regularizer_l1": [0.1, 0.0, 0.1],
	# 	"bias_regularizer_l2": [0.1, 0.0, 0.1],
	# 	"recurrent_regularizer_l1": [0.1, 0.0, 0.1],
	# 	"recurrent_regularizer_l2": [0.1, 0.0, 0.1],
	# 	"activity_regularizer_l1": [0.1, 0.0, 0.1],
	# 	"activity_regularizer_l2": [0.1, 0.0, 0.1],
	# 	"bias_initializer":[null, null, null], 
	# 	"kernel_initializer":[null, null, null]
	# }]
	def architecture(self, architectures):

		for arch in architectures:
	
			self._validate_arch_config(arch)

			model = Sequential()

			for i, unit in enumerate(arch['units']):

				kwargs = {}

				if i == 0:
					kwargs['input_shape'] = (arch['window_size'], arch['input_size'])

				# USE BIAS
				if 'use_bias' in arch and not arch['use_bias'][i]:
					kwargs['use_bias'] = False

				# REGULARIZERS -------------------------------------------------
				# kernel l1
				if 'kernel_regularizer_l1' in arch and arch['kernel_regularizer_l1'][i]:
					# both l1 and l2
					if 'kernel_regularizer_l2' in arch and arch['kernel_regularizer_l2'][i]:
						kwargs['kernel_regularizer'] = \
						l1_l2(l1=arch['kernel_regularizer_l1'][i], 
							  l2=arch['kernel_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['kernel_regularizer_l1'][i])
				# kernel l2
				elif 'kernel_regularizer_l2' in arch and arch['kernel_regularizer_l2'][i]:
					kwargs['kernel_regularizer'] = \
					    l1_l2(l2=arch['kernel_regularizer_l2'][i])

				# bias l1
				if 'bias_regularizer_l1' in arch and arch['bias_regularizer_l1'][i]:
					# both l1 and l2
					if 'bias_regularizer_l2' in arch and arch['bias_regularizer_l2'][i]:
						kwargs['bias_regularizer'] = \
						l1_l2(l1=arch['bias_regularizer_l1'][i], 
							  l2=arch['bias_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['bias_regularizer_l1'][i])
				# bias l2
				elif 'bias_regularizer_l2' in arch and arch['bias_regularizer_l2'][i]:
					kwargs['bias_regularizer'] = \
					    l1_l2(l2=arch['bias_regularizer_l2'][i]) 

				# activity l1
				if 'activity_regularizer_l1' in arch and arch['activity_regularizer_l1'][i]:
					# both l1 and l2
					if 'activity_regularizer_l2' in arch and arch['activity_regularizer_l2'][i]:
						kwargs['activity_regularizer'] = \
						l1_l2(l1=arch['activity_regularizer_l1'][i], 
							  l2=arch['activity_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['activity_regularizer_l1'][i])
				# activity l2
				elif 'activity_regularizer_l2' in arch and arch['activity_regularizer_l2'][i]:
					kwargs['activity_regularizer'] = \
					    l1_l2(l2=arch['activity_regularizer_l2'][i]) 

				if unit == 'LSTM' or unit == 'GRU':
					# recurrent l1
					if 'recurrent_regularizer_l1' in arch and arch['recurrent_regularizer_l1'][i]:
						# both l1 and l2
						if 'recurrent_regularizer_l2' in arch and arch['recurrent_regularizer_l2'][i]:
							kwargs['recurrent_regularizer'] = \
							l1_l2(l1=arch['recurrent_regularizer_l1'][i], 
								  l2=arch['recurrent_regularizer_l2'][i])
						else:
							l1_l2(l1=arch['recurrent_regularizer_l1'][i])
					# recurrent l2
					elif 'recurrent_regularizer_l2' in arch and arch['recurrent_regularizer_l2'][i]:
						kwargs['recurrent_regularizer'] = \
						    l1_l2(l2=arch['recurrent_regularizer_l2'][i]) 
					
				#---------------------------------------------------------------

				# LAYERS AND UNITS
				if unit == 'LSTM' or unit == 'GRU':
					kwargs['return_sequences'] = True \
						if 'LSTM' in arch['units'][i + 1:] or \
						'GRU' in arch['units'][i + 1:]  \
						else False

				if unit == 'LSTM':
					model.add(LSTM(arch['layers'][i], **kwargs))
				elif unit == 'GRU':
					model.add(GRU(arch['layers'][i], **kwargs))
				elif unit == 'Dense':
					model.add(Dense(arch['layers'][i]))

				# ACTIVATIONS
				if 'activations' in arch and arch['activations'][i]:
					model.add(Activation(arch['activations'][i]))

				# DROPOUT
				if 'dropout' in arch and arch['dropout'][i]:
					model.add(Dropout(arch['dropout'][i]))

			self.models.append(model)
		self.ready = True

	def _validate_arch_config(self, arch):

		def check_entries_mismatch(key, num_expected_entries):
			if key in arch and len(arch[key]) != num_expected_entries:
				raise Exception('{} and layers must have the same number of '\
				                'entries in architecture config'.format(key))

		def type_check(key, _type):
			if key in arch:
				for val in arch[key]:
					if type(val) != _type:
						raise Exception('type mismatch in {} in architecture'\
						                ' config. Expected {} got {}'\
						                .format(key, _type, type(val)))

		if not 'window_size' in arch:
			raise Exception('window_size must be included in architecture config')

		if not 'input_size' in arch:
			raise Exception('input_size must be included in architecture config')

		if not 'units' in arch:
			raise Exception('units must be included in architecture config')

		if not 'layers' in arch:
			raise Exception('units must be included in architecture config')

		if not len(arch['layers']) > 1:
			raise Exception('layers must have at least two elements')

		num_layers = len(arch['layers'])

		if len(arch['units']) != num_layers:
			raise Exception('units and layers must have the same number of '\
				            'entries in architecture config')

		check_entries_mismatch('activations', num_layers)
		check_entries_mismatch('dropout', num_layers)
		check_entries_mismatch('use_bias', num_layers)
		check_entries_mismatch('kernel_regularizer_l1', num_layers)
		check_entries_mismatch('kernel_regularizer_l2', num_layers)
		check_entries_mismatch('bias_regularizer_l1', num_layers)
		check_entries_mismatch('bias_regularizer_l2', num_layers)
		check_entries_mismatch('recurrent_regularizer_l1', num_layers)
		check_entries_mismatch('recurrent_regularizer_l2', num_layers)
		check_entries_mismatch('activity_regularizer_l1', num_layers)
		check_entries_mismatch('activity_regularizer_l2', num_layers)
		check_entries_mismatch('bias_initializer', num_layers)
		check_entries_mismatch('kernel_initializer', num_layers)

		if type(arch['window_size']) != int:
			 raise Exception('type mismatch of in {} in architecture'\
						                ' config. Expected {} got {}'\
						                .format('window_size', int, type(val)))

		if type(arch['input_size']) != int:
			 raise Exception('type mismatch of in {} in architecture'\
						                ' config. Expected {} got {}'\
						                .format('input_size', int, type(val)))

		type_check('units', str)
		type_check('layers', int)
		type_check('activations', str)
		type_check('dropout', float)
		type_check('use_bias', bool)
		type_check('kernel_regularizer_l1', float)
		type_check('kernel_regularizer_l2', float)
		type_check('bias_regularizer_l1', float)
		type_check('bias_regularizer_l2', float)
		type_check('recurrent_regularizer_l1', float)
		type_check('recurrent_regularizer_l2', float)
		type_check('activity_regularizer_l1', float)
		type_check('activity_regularizer_l2', float)
		type_check('bias_initializer', str)
		type_check('kernel_initializer', str)

		for unit in arch['units']:
			if unit not in ['LSTM', 'GRU', 'Dense']:
				raise Exception('{} is not a valid unit type')

