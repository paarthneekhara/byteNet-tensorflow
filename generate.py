import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import generator
import utils
import shutil
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sample_size', type=int, default=300,
                       help='Sampled output size')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Sample from top k predictions')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--text_dir', type=str, default='Data/generator_training_data',
                       help='Directory containing text files')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data Directory')
    parser.add_argument('--seed', type=str, default='All',
                       help='Seed for text generation')
    


    args = parser.parse_args()
    
    # model_config = json.loads( open('model_config.json').read() )
    config = model_config.predictor_config

    dl = data_loader.Data_Loader({'model_type' : 'generator', 'dir_name' : args.text_dir})
    _, vocab = dl.load_generator_data(config['sample_size'])
    
    
    model_options = {
        'vocab_size' : len(vocab),
        'residual_channels' : config['residual_channels'],
        'dilations' : config['dilations'],
        'filter_width' : config['filter_width'],
    }

    generator_model = generator.ByteNet_Generator( model_options )
    generator_model.build_generator()
    

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    
    if args.model_path:
        saver.restore(sess, args.model_path)

    seed_sentence = np.array([dl.string_to_indices(args.seed, vocab)], dtype = 'int32' )

    for col in range(args.sample_size):
        [probs] = sess.run([generator_model.g_probs], 
            feed_dict = {
                generator_model.seed_sentence :seed_sentence 
            })

        curr_preds = []
        for bi in range(probs.shape[0]):
            pred_word = utils.sample_top(probs[bi][-1], top_k = args.top_k )
            curr_preds.append(pred_word)

        seed_sentence = np.insert(seed_sentence, seed_sentence.shape[1], curr_preds, axis = 1)
        print col, dl.inidices_to_string(seed_sentence[0], vocab)

        f = open('Data/generator_sample.txt', 'wb')
        f.write(dl.inidices_to_string(seed_sentence[0], vocab))
        f.close()

if __name__ == '__main__':
    main()