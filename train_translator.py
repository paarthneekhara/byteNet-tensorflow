import argparse
import tensorflow as tf

import data_loader
import model_config
from ByteNet import model

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=0.001,
					   help='Learning Rate')
	parser.add_argument('--batch_size', type=int, default=16,
					   help='Learning Rate')
	parser.add_argument('--bucket_quant', type=int, default=25,
					   help='Learning Rate')
	parser.add_argument('--max_epochs', type=int, default=1000,
					   help='Max Epochs')
	parser.add_argument('--beta1', type=float, default=0.5,
					   help='Momentum for Adam Update')
	parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')
	parser.add_argument('--source_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.de',
                       help='Source File')
	parser.add_argument('--target_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.en',
                       help='Target File')



	args = parser.parse_args()
	
	data_loader_options = {
		'model_type' : 'translation',
		'source_file' : args.source_file,
		'target_file' : args.target_file,
		'bucket_quant' : args.bucket_quant,
	}

	dl = data_loader.Data_Loader(data_loader_options)
	buckets, source_vocab, target_vocab, frequent_keys = dl.load_translation_data()

	config = model_config.translator_config
	

	model_options = {
		'n_source_quant' : len(source_vocab),
		'n_target_quant' : len(target_vocab),
		'residual_channels' : config['residual_channels'],
		'decoder_dilations' : config['decoder_dilations'],
		'encoder_dilations' : config['encoder_dilations'],
		'sample_size' : 10,
		'decoder_filter_width' : config['decoder_filter_width'],
		'encoder_filter_width' : config['encoder_filter_width'],
		'batch_size' : args.batch_size,
		'source_mask_chars' : [ source_vocab['padding'] ],
		'target_mask_chars' : [ target_vocab['padding'] ]
	}

	

	last_saved_model_path = None
	if args.resume_model:
		last_saved_model_path = args.resume_model

	
	print("Number Of Buckets", len(buckets))

	for i in range(1, args.max_epochs):
		cnt = 0
		for _, key in frequent_keys:
			cnt += 1
			

			print("KEY", cnt, key)
			if key > 400:
				continue
			
			if len(buckets[key]) < args.batch_size:
				print("BUCKET TOO SMALL", key)
				continue

			sess = tf.InteractiveSession()
			
			

			batch_no = 0
			batch_size = args.batch_size

			byte_net = model.Byte_net_model( model_options )
			bn_tensors = byte_net.build_translation_model(sample_size = key)
			
			adam = tf.train.AdamOptimizer(
				args.learning_rate, 
				beta1 = args.beta1)

			optim = adam.minimize(bn_tensors['loss'], var_list=bn_tensors['variables'])
			
			
			train_writer = tf.train.SummaryWriter('logs/', sess.graph)
			tf.initialize_all_variables().run()

			saver = tf.train.Saver()
			if last_saved_model_path:
				saver.restore(sess, last_saved_model_path)

			while (batch_no + 1) * batch_size < len(buckets[key]):
				source, target = dl.get_batch_from_pairs( 
					buckets[key][batch_no * batch_size : (batch_no+1) * batch_size] 
				)

				_, loss, prediction, summary, source_gradient, target_gradient = sess.run( 
					[optim, bn_tensors['loss'], bn_tensors['prediction'],
					 bn_tensors['merged_summary'], bn_tensors['source_gradient'], bn_tensors['target_gradient']], 
					feed_dict = {
						bn_tensors['source_sentence'] : source,
						bn_tensors['target_sentence'] : target,
					})
				
				train_writer.add_summary(summary, batch_no * (cnt + 1))
				print("Loss", loss, batch_no, len(buckets[key])/batch_size, i, cnt, key)
				
				print("******")
				print("Source ", dl.inidices_to_string(source[0], source_vocab))
				print("---------")
				print("Target ", dl.inidices_to_string(target[0], target_vocab))
				print("----------")
				print("Prediction ",dl.inidices_to_string(prediction[0:key], target_vocab))
				print("******")
				
				batch_no += 1
				if batch_no % 1000 == 0:
					last_saved_model_path = "Data/Models/model_translation_epoch_{}_{}.ckpt".format(i, cnt)
					saver.save(sess, last_saved_model_path)

			last_saved_model_path = "Data/Models/model_translation_epoch_{}.ckpt".format(i)
			saver.save(sess, last_saved_model_path)

			tf.reset_default_graph()
			sess.close()

if __name__ == '__main__':
	main()