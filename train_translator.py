import tensorflow as tf
import numpy as np
import argparse
import model_config
import data_loader
from ByteNet import translator
import utils
import shutil
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Learning Rate')
    parser.add_argument('--bucket_quant', type=int, default=50,
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
    parser.add_argument('--sample_every', type=int, default=500,
                       help='Sample generator output evry x steps')
    parser.add_argument('--summary_every', type=int, default=50,
                       help='Sample generator output evry x steps')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Sample from top k predictions')
    parser.add_argument('--resume_from_bucket', type=int, default=0,
                       help='Resume From Bucket')
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
    translator_model.build_model()

    optim = tf.train.AdamOptimizer(
        args.learning_rate, 
        beta1 = args.beta1).minimize(translator_model.loss)

    translator_model.build_translator(reuse = True)
    merged_summary = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    if args.resume_model:
        saver.restore(sess, args.resume_model)

    shutil.rmtree('Data/tb_summaries/translator_model')
    train_writer = tf.summary.FileWriter('Data/tb_summaries/translator_model', sess.graph)
    
    bucket_sizes = [bucket_size for bucket_size in buckets]
    bucket_sizes.sort()

    step = 0
    batch_size = args.batch_size
    for epoch in range(args.max_epochs):
        for bucket_size in bucket_sizes:
            if epoch == 0 and bucket_size < args.resume_from_bucket:
                continue

            batch_no = 0
            while (batch_no + 1) * batch_size < len(buckets[bucket_size]):
                start = time.clock()
                source, target = dl.get_batch_from_pairs( 
                    buckets[bucket_size][batch_no * batch_size : (batch_no+1) * batch_size] 
                )
                
                _, loss, prediction = sess.run( 
                    [optim, translator_model.loss, translator_model.arg_max_prediction], 
                    
                    feed_dict = {
                        translator_model.source_sentence : source,
                        translator_model.target_sentence : target,
                    })
                end = time.clock()

                print "LOSS: {}\tEPOCH: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}\t bucket_size:{}".format(
                loss, epoch, batch_no, step, len(buckets[bucket_size])/args.batch_size, bucket_size)
                print "TIME FOR BATCH", end - start
                print "TIME FOR BUCKET (mins)", (end - start) * (len(buckets[bucket_size])/args.batch_size)/60.0

                batch_no += 1
                step += 1

                if step % args.summary_every == 0:
                    [summary] = sess.run([merged_summary], feed_dict = {
                        translator_model.source_sentence : source,
                        translator_model.target_sentence : target,
                    })
                    train_writer.add_summary(summary, step)

                    print "******"
                    print "Source ", dl.inidices_to_string(source[0], source_vocab)
                    print "---------"
                    print "Target ", dl.inidices_to_string(target[0], target_vocab)
                    print "----------"
                    print "Prediction ",dl.inidices_to_string(prediction[0:bucket_size], target_vocab)
                    print "******"

                if step % args.sample_every == 0:
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
                                print "***************"
                    log_file.close()

            save_path = saver.save(sess, "Data/Models/translation_model/model_epoch_{}_{}.ckpt".format(epoch, bucket_size))



if __name__ == '__main__':
    main()



