import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader_v2
from ByteNet import model
import utils

# STILL IN DEVELOPMENT...
def main():
	parser = argparse.ArgumentParser()
	
	
	parser.add_argument('--model_path', type=str, default='Data/Models/model_translation_epoch_1.ckpt',
                       help='Pre-Trained Model Path')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data Directory')
	parser.add_argument('--num_char', type=int, default=1000,
                       help='seed')
	parser.add_argument('--translator_max_length', type=int, default=500,
                       help='translator_max_length')

	parser.add_argument('--output_file', type=str, default='sample.txt',
                       help='Output File')


	args = parser.parse_args()
	
	
	
	config = model_config.translator_config

	source_sentence = None
	with open('Data/MachineTranslation/news-commentary-v11.de-en.de') as f:
		source_sentences = f.read().decode("utf-8").split('\n')

	with open('Data/MachineTranslation/news-commentary-v11.de-en.en') as f:
		target_sentences = f.read().decode("utf-8").split('\n')

	idx = 0
	for i in range(len(source_sentences)):
		if 'NEW YORK' in target_sentences[i][0:40]:
			print target_sentences[i]
			idx = i
			break

	source_sentences = source_sentences[idx : idx + 1]
	target_sentences = target_sentences[idx : idx + 1]

	print source_sentences
	print target_sentences

	data_loader_options = {
		'model_type' : 'translation',
		'source_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.de',
		'target_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.en',
		'bucket_quant' : 25,
	}

	dl = data_loader_v2.Data_Loader(data_loader_options)
	# buckets, source_vocab, target_vocab, frequent_keys = dl.load_translation_data()

	source_ = []
	target_ = []
	for i in range(len(source_sentences)):
		source_sentence = source_sentences[i]

		source = [ dl.source_vocab[s] for s in source_sentence ]
		source += [ dl.source_vocab['eol'] ]

		new_length = args.translator_max_length
		# bucket_quant = args.bucket_quant
		# if new_length % bucket_quant > 0:
		# 	new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant

		for i in range(len(source), new_length):
			source += [ dl.source_vocab['padding'] ]

		target = [ dl.target_vocab['init'] ]
		for j in range(1, new_length + 1):
			target += [ dl.target_vocab['padding'] ]

		source_.append(source)
		target_.append(target)

	
	source = np.array(source_)
	target = np.array(target_)
	# print source_
	# source = np.array(source_, dtype='int32')
	# target = np.array(target_, dtype='int32')
	
	# print source
	# print target

	model_options = {
		'n_source_quant' : len(dl.source_vocab),
		'n_target_quant' : len(dl.target_vocab),
		'residual_channels' : config['residual_channels'],
		'decoder_dilations' : config['decoder_dilations'],
		'encoder_dilations' : config['encoder_dilations'],
		'sample_size' : 10,
		'decoder_filter_width' : config['decoder_filter_width'],
		'encoder_filter_width' : config['encoder_filter_width'],
		'batch_size' : 1,
		'source_mask_chars' : [ dl.source_vocab['padding'] ],
		'target_mask_chars' : [ dl.target_vocab['padding'] ]
	}

	byte_net = model.Byte_net_model( model_options )
	translator = byte_net.build_translation_model( args.translator_max_length )
	
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)

	input_batch = target
	print "INPUT", input_batch
	print "Source", source
	
	for i in range(0, 1000):
		
		prediction, probs = sess.run( 
			[translator['prediction'], translator['probs']], 
			feed_dict = {
				translator['source_sentence'] : source,
				translator['target_sentence'] : input_batch,
				})
		# prediction = prediction[0]
		last_prediction = np.array( [  utils.weighted_pick( probs[i] ) ])
		last_prediction = last_prediction.reshape([1,-1])
		# prediction = np.reshape(prediction, )
	# 	print "encoder"
	# 	print encoder_output
	# 	last_prediction =  prediction[i]
		
	# 	last_prediction = np.array( [  last_prediction ])
		
	# 	last_prediction = last_prediction.reshape([1,-1])
		input_batch[:,i+1] = last_prediction[:,0]
		res = dl.inidices_to_string(input_batch[0], dl.target_vocab)
		print "RES"
		print res
		
def weighted_pick(weights):
	t = np.cumsum(weights)
	s = np.sum(weights)
	return(int(np.searchsorted(t, np.random.rand(1)*s)))


def list_to_string(ascii_list):
	res = u""
	for a in ascii_list:
		if a >= 0 and a < 256:
			res += unichr(a)
	return res

if __name__ == '__main__':
	main()

