import tensorflow as tf
import ops

class ByteNet_Translator:
    def __init__(self, options):
        self.options = options
        embedding_channels = 2 * options['residual_channels']

        self.w_source_embedding = tf.get_variable('w_source_embedding', 
                    [options['source_vocab_size'], embedding_channels],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.w_target_embedding = tf.get_variable('w_target_embedding', 
                    [options['target_vocab_size'], embedding_channels],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    def build_model(self):
        options = self.options
        self.source_sentence = tf.placeholder('int32', 
            [None, None], name = 'source_sentence')
        self.target_sentence = tf.placeholder('int32', 
            [None, None], name = 'target_sentence')

        target_1 = self.target_sentence[:,0:-1]
        target_2 = self.target_sentence[:,1:]

        source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, 
            self.source_sentence, name = "source_embedding")
        target_1_embedding = tf.nn.embedding_lookup(self.w_target_embedding, 
            target_1, name = "target_1_embedding")


        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['encoder_dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['encoder_filter_width'], causal = False, train = True)

        encoder_output = curr_input
        combined_embedding = target_1_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(options['decoder_dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['decoder_filter_width'], causal = True, train = True)

        logits = ops.conv1d(tf.nn.relu(curr_input), 
            options['target_vocab_size'], name = 'logits')
        print "logits", logits
        logits_flat = tf.reshape(logits, [-1, options['target_vocab_size']])
        target_flat = tf.reshape(target_2, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = target_flat, logits = logits_flat)
        
        self.loss = tf.reduce_mean(loss)
        self.arg_max_prediction = tf.argmax(logits_flat, 1)
        tf.summary.scalar('loss', self.loss)

    def build_translator(self, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        options = self.options
        self.t_source_sentence = tf.placeholder('int32', 
            [None, None], name = 'source_sentence')
        self.t_target_sentence = tf.placeholder('int32', 
            [None, None], name = 'target_sentence')

        source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, 
            self.t_source_sentence, name = "source_embedding")
        target_embedding = tf.nn.embedding_lookup(self.w_target_embedding, 
            self.t_target_sentence, name = "target_embedding")

        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['encoder_dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['encoder_filter_width'], causal = False, train = False)

        encoder_output = curr_input[:,0:tf.shape(self.t_target_sentence)[1],:]

        combined_embedding = target_embedding + encoder_output
        curr_input = combined_embedding
        for layer_no, dilation in enumerate(options['decoder_dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['decoder_filter_width'], causal = True, train = False)

        logits = ops.conv1d(tf.nn.relu(curr_input), 
            options['target_vocab_size'], name = 'logits')
        logits_flat = tf.reshape(logits, [-1, options['target_vocab_size']])
        probs_flat = tf.nn.softmax(logits_flat)

        self.t_probs = tf.reshape(probs_flat, 
            [-1, tf.shape(logits)[1], options['target_vocab_size']])

def main():
    options = {
        'source_vocab_size' : 250,
        'target_vocab_size' : 250,
        'residual_channels' : 512,
        'encoder_dilations' : [ 1,2,4,8,16,
                        1,2,4,8,16
                       ],
        'decoder_dilations' : [ 1,2,4,8,16,
            1,2,4,8,16
        ],
        'encoder_filter_width' : 3,
        'decoder_filter_width' : 3
    }
    md = ByteNet_Translator(options)
    md.build_model()
    md.build_translator(reuse = True)

if __name__ == '__main__':
    main()