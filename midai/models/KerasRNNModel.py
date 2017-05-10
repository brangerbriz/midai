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
	def architecture(self, architecture):

		for arch in architecture:
	
			self._validate_arch_config(arch)

			model = Sequential()

			for i, unit in enumerate(arch['units']):

				kwargs = {}

				if i == 0:
					kwargs['input_shape'] = (arch['window_size'], arch['input_size'])

				# USE BIAS
				if not arch['use_bias'][i]:
					kwargs['use_bias'] = False

				# REGULARIZERS -------------------------------------------------
				# kernel l1
				if arch['kernel_regularizer_l1'][i]:
					# both l1 and l2
					if arch['kernel_regularizer_l2'][i]:
						kwargs['kernel_regularizer'] = \
						l1_l2(l1=arch['kernel_regularizer_l1'][i], 
							  l2=arch['kernel_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['kernel_regularizer_l1'][i])
				# kernel l2
				elif arch['kernel_regularizer_l2'][i]:
					kwargs['kernel_regularizer'] = \
					    l1_l2(l2=arch['kernel_regularizer_l1'][i])

				# bias l1
				if arch['bias_regularizer_l1'][i]:
					# both l1 and l2
					if arch['bias_regularizer_l2'][i]:
						kwargs['bias_regularizer'] = \
						l1_l2(l1=arch['bias_regularizer_l1'][i], 
							  l2=arch['bias_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['bias_regularizer_l1'][i])
				# bias l2
				elif arch['bias_regularizer_l2'][i]:
					kwargs['bias_regularizer'] = \
					    l1_l2(l2=arch['bias_regularizer_l1'][i]) 

				# activity l1
				if arch['activity_regularizer_l1'][i]:
					# both l1 and l2
					if arch['activity_regularizer_l2'][i]:
						kwargs['activity_regularizer'] = \
						l1_l2(l1=arch['activity_regularizer_l1'][i], 
							  l2=arch['activity_regularizer_l2'][i])
					else:
						l1_l2(l1=arch['activity_regularizer_l1'][i])
				# activity l2
				elif arch['activity_regularizer_l2'][i]:
					kwargs['activity_regularizer'] = \
					    l1_l2(l2=arch['activity_regularizer_l1'][i]) 

				if unit == 'LSTM' or unit == 'GRU':
					# recurrent l1
					if arch['recurrent_regularizer_l1'][i]:
						# both l1 and l2
						if arch['recurrent_regularizer_l2'][i]:
							kwargs['recurrent_regularizer'] = \
							l1_l2(l1=arch['recurrent_regularizer_l1'][i], 
								  l2=arch['recurrent_regularizer_l2'][i])
						else:
							l1_l2(l1=arch['recurrent_regularizer_l1'][i])
					# recurrent l2
					elif arch['recurrent_regularizer_l2'][i]:
						kwargs['recurrent_regularizer'] = \
						    l1_l2(l2=arch['recurrent_regularizer_l1'][i]) 
					
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
				if arch['activations'][i]:
					model.add(Activation(arch['activations'][i]))

				# DROPOUT
				if arch['dropout'][i]:
					model.add(Dropout(arch['dropout'][i]))

			self.models.append(model)
		self.ready = True
	
	def _validate_arch_config(self, arch):

		# units
		if not arch['units'] or not len(arch['units']) > 1:
			raise Exception('{} is not a valid units entry'.format(arch.units))

		for unit in arch['units']:
			if unit not in ['LSTM', 'GRU', 'Dense']:
				raise Exception('{} is not a valid unit type')

