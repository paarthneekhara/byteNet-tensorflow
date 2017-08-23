import tensorflow as tf
import ops

class ByteNet_Generator:
    def __init__(self, options):
        self.options = options
        source_embedding_channels = 2 * options['residual_channels']
        self.w_sentence_embedding = tf.get_variable('w_sentence_embedding', 
                [options['vocab_size'], source_embedding_channels],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

    def build_model(self):
        options = self.options
        self.t_sentence = tf.placeholder('int32', 
            [None, None], name = 't_sentence')

        source_sentence = self.t_sentence[:,0:-1]
        target_sentence = self.t_sentence[:,1:]

        source_embedding = tf.nn.embedding_lookup(self.w_sentence_embedding, 
            source_sentence, name = "source_embedding")

        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['filter_width'], causal = True, train = True)

        logits = ops.conv1d(tf.nn.relu(curr_input), 
            options['vocab_size'], name = 'logits')

        logits_flat = tf.reshape(logits, [-1, options['vocab_size']])
        target_flat = tf.reshape(target_sentence, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_flat, logits = logits_flat)
        self.loss = tf.reduce_mean(loss)
        
        self.arg_max_prediction = tf.argmax(logits_flat, 1)
        
        tf.summary.scalar('loss', self.loss)

    def build_generator(self, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        options = self.options
        self.seed_sentence = tf.placeholder('int32', 
            [None, None], name = 'seed_sentence')
        
        source_embedding = tf.nn.embedding_lookup(self.w_sentence_embedding, 
            self.seed_sentence, name = "source_embedding")

        curr_input = source_embedding
        for layer_no, dilation in enumerate(options['dilations']):
            curr_input = ops.byetenet_residual_block(curr_input, dilation, 
                layer_no, options['residual_channels'], 
                options['filter_width'], causal = True, train = False)

        logits = ops.conv1d(tf.nn.relu(curr_input), 
            options['vocab_size'], name = 'logits')
        logits_flat = tf.reshape(logits, [-1, options['vocab_size']])
        probs_flat = tf.nn.softmax(logits_flat)
        
        self.g_probs = tf.reshape(probs_flat, [-1, tf.shape(self.seed_sentence)[1], options['vocab_size']])
        

def main():
    options = {
        'vocab_size' : 250,
        'residual_channels' : 512,
        'dilations' : [ 1,2,4,8,16,
                        1,2,4,8,16
                       ],
        'filter_width' : 3
    }

    model = ByteNet_Generator(options)
    model.build_model()
    model.build_generator(reuse = True)

if __name__ == '__main__':
    main()