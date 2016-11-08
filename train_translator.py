import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=0.001,
					   help='Learning Rate')
	parser.add_argument('--batch_size', type=int, default=1,
					   help='Learning Rate')
	parser.add_argument('--max_epochs', type=int, default=1000,
					   help='Max Epochs')
	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')
	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')
	parser.add_argument('--source_file', type=str, default='Data/MachineTranslation/europarl-v7.de-en.de',
                       help='Source File')
	parser.add_argument('--target_file', type=str, default='Data/MachineTranslation/europarl-v7.de-en.en',
                       help='Target File')
	


	args = parser.parse_args()
	
	# model_config = json.loads( open('model_config.json').read() )
	
	config = model_config.translator_config

	model_options = {
		'n_source_quant' : config['n_source_quant'],
		'n_target_quant' : config['n_target_quant'],
		'residual_channels' : config['residual_channels'],
		'decoder_dilations' : config['decoder_dilations'],
		'encoder_dilations' : config['encoder_dilations'],
		'sample_size' : 10,
		'decoder_filter_width' : config['decoder_filter_width'],
		'encoder_filter_width' : config['encoder_filter_width'],
		'batch_size' : args.batch_size,
	}

	byte_net = model.Byte_net_model( model_options )
	bn_tensors = byte_net.build_translation_model(model_options['sample_size'])

	optim = tf.train.AdamOptimizer(
				args.learning_rate, 
				beta1 = args.beta1).minimize(bn_tensors['loss'], var_list=bn_tensors['variables'])

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	
	saver = tf.train.Saver()

	if args.resume_model:
		saver.restore(sess, args.resume_model)

	buckets, freq_keys = data_loader.load_translation_data(args.source_file, args.target_file)
	print len(buckets)

	tf.get_variable_scope().reuse_variables()
	for i in range(args.max_epochs):
		for _, key in freq_keys:
			batch_no = 0
			batch_size = args.batch_size
			
			bn_tensors = byte_net.build_translation_model(sample_size = key)
			
			adam = tf.train.AdamOptimizer(
				args.learning_rate, 
				beta1 = args.beta1)

			optim = adam.minimize(bn_tensors['loss'], var_list=bn_tensors['variables'])
			
			print "CHECK"
			uninitialized_variables = list( str(name) for name in sess.run( tf.report_uninitialized_variables( tf.all_variables( ) ) ) )	
			
			all_variables = tf.all_variables()

			uninitialized = []
			for av in all_variables:
				for uv in uninitialized_variables:
					if uv in av.name:
						uninitialized.append(av)

			print "CHECK@", len(uninitialized)
			tf.initialize_variables( uninitialized ).run()

			while (batch_no + 1) * batch_size < len(buckets[key]):
				source, target = data_loader.get_batch_from_pairs( 
					buckets[key][batch_no * batch_size : (batch_no+1) * batch_size] 
				)

				_, loss, prediction = sess.run( [optim, bn_tensors['loss'], bn_tensors['prediction']], feed_dict = {
					bn_tensors['source_sentence'] : source,
					bn_tensors['target_sentence'] : target,
				})

				print "Loss", loss, batch_no, len(buckets[key])/batch_size, i
				print "prediction"
				print list_to_string(prediction)
				batch_no += 1
				if batch_no % 500 == 0:
					save_path = saver.save(sess, "Data/Models/model_translation_epoch_{}.ckpt".format(i))
				


		

		# batch_no = 0
		# batch_size = args.batch_size
		# while (batch_no+1) * batch_size < text_samples.shape[0]:
		# 	text_batch = text_samples[batch_no*batch_size : (batch_no + 1)*batch_size, :]
		# 	_, loss, prediction = sess.run( [optim, bn_tensors['loss'], bn_tensors['prediction']], feed_dict = {
		# 		bn_tensors['sentence'] : text_batch
		# 		})
		# 	print "-------------------------------------------------------"
		# 	print list_to_string(prediction)
		# 	print "Loss"

		# 	print i, batch_no, loss
		# 	print "********************************************************"
		# 	# print prediction
		# 	batch_no += 1
			
		# 	if (batch_no % 100) == 0:
		# 		save_path = saver.save(sess, "Data/Models/model_epoch_{}.ckpt".format(i))

def list_to_string(ascii_list):
	res = u""
	for a in ascii_list:
		if a >= 0 and a < 256:
			res += unichr(a)
	return res

if __name__ == '__main__':
	main()