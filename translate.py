import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import translator
import utils
import shutil
import time
import random

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bucket_quant', type=int, default=50,
                       help='Learning Rate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--source_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.de',
                       help='Source File')
    parser.add_argument('--target_file', type=str, default='Data/MachineTranslation/news-commentary-v11.de-en.en',
                       help='Target File')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Sample from top k predictions')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch Size')
    parser.add_argument('--bucket_size', type=int, default=None,
                       help='Bucket Size')
    args = parser.parse_args()
    
    data_loader_options = {
        'model_type' : 'translation',
        'source_file' : args.source_file,
        'target_file' : args.target_file,
        'bucket_quant' : args.bucket_quant,
    }

    dl = data_loader.Data_Loader(data_loader_options)
    buckets, source_vocab, target_vocab = dl.load_translation_data()
    print "Number Of Buckets", len(buckets)

    config = model_config.translator_config
    model_options = {
        'source_vocab_size' : len(source_vocab),
        'target_vocab_size' : len(target_vocab),
        'residual_channels' : config['residual_channels'],
        'decoder_dilations' : config['decoder_dilations'],
        'encoder_dilations' : config['encoder_dilations'],
        'decoder_filter_width' : config['decoder_filter_width'],
        'encoder_filter_width' : config['encoder_filter_width'],
    }

    translator_model = translator.ByteNet_Translator( model_options )
    translator_model.build_translator()
    
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    if args.model_path:
        saver.restore(sess, args.model_path)

    
    
    bucket_sizes = [bucket_size for bucket_size in buckets]
    bucket_sizes.sort()

    if not args.bucket_size:
        bucket_size = random.choice(bucket_sizes)
    else:
        bucket_size = args.bucket_size

    source, target = dl.get_batch_from_pairs( 
        random.sample(buckets[bucket_size], args.batch_size)
    )
    
    log_file = open('Data/translator_sample.txt', 'wb')
    generated_target = target[:,0:1]
    for col in range(bucket_size):
        [probs] = sess.run([translator_model.t_probs], 
            feed_dict = {
                translator_model.t_source_sentence : source,
                translator_model.t_target_sentence : generated_target,
            })

        curr_preds = []
        for bi in range(probs.shape[0]):
            pred_word = utils.sample_top(probs[bi][-1], top_k = args.top_k )
            curr_preds.append(pred_word)

        generated_target = np.insert(generated_target, generated_target.shape[1], curr_preds, axis = 1)
        

        for bi in range(probs.shape[0]):

            print col, dl.inidices_to_string(generated_target[bi], target_vocab)
            print col, dl.inidices_to_string(target[bi], target_vocab)
            print "***************"

            if col == bucket_size - 1:
                try:
                    log_file.write("Predicted: " + dl.inidices_to_string(generated_target[bi], target_vocab) + '\n')
                    log_file.write("Actual Target: " + dl.inidices_to_string(target[bi], target_vocab) + '\n')
                    log_file.write("Actual Source: " + dl.inidices_to_string(source[bi], source_vocab) + '\n *******')
                except:
                    pass
                
    log_file.close()



if __name__ == '__main__':
    main()