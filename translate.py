import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import model


# STILL IN DEVELOPMENT...
def main():
	parser = argparse.ArgumentParser()
	
	
	parser.add_argument('--model_path', type=str, default='Data/Models/model_translation_epoch_1.ckpt',
                       help='Pre-Trained Model Path')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data Directory')
	parser.add_argument('--num_char', type=int, default=1000,
                       help='seed')

	parser.add_argument('--output_file', type=str, default='sample.txt',
                       help='Output File')


	args = parser.parse_args()
	
	
	
	config = model_config.translator_config

	source_sentence = None
	with open('Data/MachineTranslation/news-commentary-v11.de-en.de') as f:
		source_sentences = f.read().decode("utf-8").split('\n')

	with open('Data/MachineTranslation/news-commentary-v11.de-en.en') as f:
		target_sentences = f.read().decode("utf-8").split('\n')

	source_sentence = source_sentences[4]
	target_sentence = target_sentences[4]

	print source_sentence
	print target_sentence


	data_loader_options = {
		'model_type' : 'translation',
		'source_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.de',
		'target_file' : 'Data/MachineTranslation/news-commentary-v11.de-en.en'
		'bucket_quant' : 25,
	}

	dl = data_loader_v2.Data_Loader(data_loader_options)
	# buckets, source_vocab, target_vocab, frequent_keys = dl.load_translation_data()

	with open('source.txt', 'wb') as f:
		f.write(source_sentence.encode('utf8'))
	source = [ dl.source_vocab(s) for s in source_sentence ]
	source += [ dl.source_vocab['eol'] ]

	new_length = len(source)
	bucket_quant = data_loader.BUCKET_QUANT
	if new_length % bucket_quant > 0:
		new_length = ((new_length/bucket_quant) + 1 ) * bucket_quant

	for i in range(len(source), new_length):
		source += [ dl.source_vocab['padding'] ]

	target = [ dl.target_vocab['target_init'] ]
	for j in range(1, new_length):
		target += [ dl.target_vocab['padding'] ]

	print "SL", len(source)
	print "TL", len(target)

	source = np.array(source, dtype='int32')
	source = source.reshape([1, -1])

	target = np.array(target, dtype='int32')
	target = target.reshape([1, -1])

	model_options = {
		'n_source_quant' : len(dl.source_vocab),
		'n_target_quant' : len(dl.target_vocab),
		'residual_channels' : config['residual_channels'],
		'decoder_dilations' : config['decoder_dilations'],
		'encoder_dilations' : config['encoder_dilations'],
		'sample_size' : 10,
		'decoder_filter_width' : config['decoder_filter_width'],
		'encoder_filter_width' : config['encoder_filter_width'],
		'batch_size' : args.batch_size,
		'source_mask_chars' : [ dl.source_vocab['padding'] ],
		'target_mask_chars' : [ dl.target_vocab['padding'] ]
	}

	byte_net = model.Byte_net_model( model_options )
	translator = byte_net.build_translator( new_length )
	
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)

	input_batch = target
	print "INPUT", input_batch
	print "Source", source
	for i in range(0, 1000):
		
		prediction, encoder_output = sess.run( 
			[translator['prediction'], translator['encoder_output']], 
			feed_dict = {
				translator['source_sentence'] : source,
				translator['target_sentence'] : input_batch,
				})
		# prediction = prediction[0]

		print "encoder"
		print encoder_output
		last_prediction =  prediction[i]
		
		last_prediction = np.array( [  last_prediction ])
		
		last_prediction = last_prediction.reshape([1,-1])
		input_batch[:,i+1] = last_prediction[:,0]
		res = dl.inidices_to_string(input_batch[0], dl.target_vocab)
		print "RES"
		print res
		with open('sample.txt', 'wb') as f:
			f.write(res)

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

