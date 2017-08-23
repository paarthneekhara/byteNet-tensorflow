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
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Learning Rate')
    parser.add_argument('--sample_every', type=int, default=500,
                       help='Sample generator output evry x steps')
    parser.add_argument('--summary_every', type=int, default=50,
                       help='Sample generator output evry x steps')
    parser.add_argument('--save_model_every', type=int, default=1500,
                       help='Save model every')
    parser.add_argument('--sample_size', type=int, default=300,
                       help='Sampled output size')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Sample from top k predictions')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Max Epochs')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Momentum for Adam Update')
    parser.add_argument('--resume_model', type=str, default=None,
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
    text_samples, vocab = dl.load_generator_data(config['sample_size'])
    print text_samples.shape
    
    model_options = {
        'vocab_size' : len(vocab),
        'residual_channels' : config['residual_channels'],
        'dilations' : config['dilations'],
        'filter_width' : config['filter_width'],
    }

    generator_model = generator.ByteNet_Generator( model_options )
    generator_model.build_model()
    
    optim = tf.train.AdamOptimizer(
        args.learning_rate, 
        beta1 = args.beta1).minimize(generator_model.loss)

    generator_model.build_generator(reuse = True)
    merged_summary = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    
    if args.resume_model:
        saver.restore(sess, args.resume_model)
    
    shutil.rmtree('Data/tb_summaries/generator_model')
    train_writer = tf.summary.FileWriter('Data/tb_summaries/generator_model', sess.graph)

    step = 0
    for epoch in range(args.max_epochs):
        batch_no = 0
        batch_size = args.batch_size
        while (batch_no+1) * batch_size < text_samples.shape[0]:

            start = time.clock()

            text_batch = text_samples[batch_no*batch_size : (batch_no + 1)*batch_size, :]
            _, loss, prediction = sess.run( 
                [optim, generator_model.loss, 
                generator_model.arg_max_prediction], 
                feed_dict = {
                    generator_model.t_sentence : text_batch
                })
            end = time.clock()
            print "-------------------------------------------------------"
            print "LOSS: {}\tEPOCH: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                loss, epoch, batch_no, step, text_samples.shape[0]/args.batch_size)
            print "TIME FOR BATCH", end - start
            print "TIME FOR EPOCH (mins)", (end - start) * (text_samples.shape[0]/args.batch_size)/60.0
            
            batch_no += 1
            step += 1
            
            if step % args.summary_every == 0:
                [summary] = sess.run([merged_summary], feed_dict = {
                    generator_model.t_sentence : text_batch
                })
                train_writer.add_summary(summary, step)
                print dl.inidices_to_string(prediction, vocab)
            
            print "********************************************************"
                
            if step % args.sample_every == 0:
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

            if step % args.save_model_every == 0:
                save_path = saver.save(sess, "Data/Models/generation_model/model_epoch_{}_{}.ckpt".format(epoch, step))

if __name__ == '__main__':
    main()