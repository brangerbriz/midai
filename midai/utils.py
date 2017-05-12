import os
import inspect

# reads value from LOG_LEVEL environment variable and conditionally
# logs messages accourdingly
def log(message, level, default='VERBOSE'):

	level = level.upper()
	log_level = str(os.getenv('LOG_LEVEL', default.upper())).upper()
	log_dict = {
		"VERBOSE": 1,
		"NOTICE": 2,
		"WARNING": 3,
		"ERROR": 4,
		"FATAL": 5,
		"SILENT": 0
	}

	if level not in log_dict:
		raise Exception('{} is not a supported log level'.format(level))
	if log_level not in log_dict:
		raise Exception('{} is not a supported log level'.format(log_level))

	if log_level == 'SILENT':
		return
	elif log_dict[level] >= log_dict[log_level]: 
		print('[{}] {} | {}'.format(level.upper().center(7), 
			                        inspect.stack()[1].function,
			                        message))

def clamp(val, min_, max_):
    return min_ if val < min_ else max_ if val > max_ else val

def map_range(a, b, s):
    (a1, a2), (b1, b2) = a, b
    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))
