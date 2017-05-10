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

# tests the log function
def test_log():

	def log_test():
		print('')
		message = 'this is a test message.'
		log(message, 'verbose')
		log(message, 'notice')
		log(message, 'warning')
		log(message, 'error')
		log(message, 'fatal')
		print('')
	
	os.environ['LOG_LEVEL'] = 'VERBOSE'
	print('set log level VERBOSE')
	log_test()
	
	os.environ['LOG_LEVEL'] = 'NOTICE'
	print('set log level NOTICE')
	log_test()

	os.environ['LOG_LEVEL'] = 'WARNING'
	print('set log level WARNING')
	log_test()

	os.environ['LOG_LEVEL'] = 'ERROR'
	print('set log level ERROR')
	log_test()

	os.environ['LOG_LEVEL'] = 'FATAL'
	print('set log level FATAL')
	log_test()

if __name__ == '__main__':
	test_log()