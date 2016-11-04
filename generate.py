import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import model

def main():
	parser = argparse.ArgumentParser()
	
	
	parser.add_argument('--model_path', type=str, default=None,
                       help='Pre-Trained Model Path')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data Directory')
	parser.add_argument('--seed', type=str, default="ANTONIO",
                       help='seed')
	parser.add_argument('--num_char', type=int, default=1000,
                       help='seed')

	parser.add_argument('--output_file', type=str, default='sample.txt',
                       help='Output File')


	args = parser.parse_args()
	
	# model_config = json.loads( open('model_config.json').read() )
	
	config = model_config.config

	model_options = {
		'n_source_quant' : config['n_source_quant'],
		'n_target_quant' : config['n_target_quant'],
		'residual_channels' : config['residual_channels'],
		'decoder_dilations' : config['decoder_dilations'],
		'sample_size' : config['sample_size'],
		'decoder_filter_width' : config['decoder_filter_width'],
		'batch_size' : 1,
	}

	seed_ = [ ord(s) for s in args.seed ]
	seed_ = np.array(seed_, dtype='int32')
	seed_ = seed_.reshape([1, -1])

	byte_net = model.Byte_net_model( model_options )
	generator = byte_net.build_generator( len(args.seed) )
	
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)

	input_batch = seed_
	print "INPUT", input_batch
	for i in range(0, args.num_char):
		generator = byte_net.build_generator( input_batch.shape[1], reuse = True)
		prediction = sess.run( [generator['prediction']], 
			feed_dict = {
				generator['source_sentence'] : input_batch
				})
		prediction = prediction[0]
		
		last_prediction =  prediction[ prediction.shape[0] - 1 ]
		last_prediction = last_prediction.reshape([1,-1])
		input_batch = np.concatenate((input_batch, last_prediction), axis = 1)
		res = list_to_string(input_batch[0])
		print res
		with open(args.output_file, 'wb') as f:
			f.write(res)

def list_to_string(ascii_list):
	res = ""
	for a in ascii_list:
		res += str(chr(a))
	return res

if __name__ == '__main__':
	main()

