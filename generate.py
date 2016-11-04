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
	parser.add_argument('--seed', type=str, default='PORTIA',
                       help='seed')


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

	seed_ = [ ord(s) for s in seed ]
	seed_ = np.array(seed_, dtype='int32')
	seed_.reshape([1, -1])

	byte_net = model.Byte_net_model( model_options )
	generator = byte_net.build_generator( len(seed) )
	
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)

	input_batch = seed_
	for i in range(0, 200):
		generator = byte_net.build_generator( input_batch.shape[0] )
		prediction = sess.run( [generator['prediction']], 
			feed_dict = {
				generator['sentence'] : input_batch
				})
		print i, prediction
		last_prediction = prediction[prediction.shape[0] - 1 ]
		input_batch = np.concatenate((input_batch, last_prediction), axis = 1)




