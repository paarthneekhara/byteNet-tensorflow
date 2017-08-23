import tensorflow as tf
import math

def fully_connected(input_, output_nodes, name, stddev=0.02):
    with tf.variable_scope(name):
        input_shape = input_.get_shape()
        input_nodes = input_shape[-1]
        w = tf.get_variable('w', [input_nodes, output_nodes], 
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('b', [output_nodes], 
            initializer=tf.constant_initializer(0.0))
        res = tf.matmul(input_, w) + biases
        return res


# 1d CONVOLUTION WITH DILATION
def conv1d(input_, output_channels, 
    dilation = 1, filter_width = 1, causal = False, 
    name = "dilated_conv"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [1, filter_width, input_.get_shape()[-1], output_channels ],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_channels ],
           initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim = 1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'VALID') + b
        else:
            input_expanded = tf.expand_dims(input_, dim = 1)
            out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'SAME') + b

        return tf.squeeze(out, [1])

def layer_normalization(x, name, epsilon=1e-8, trainable = True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [ int(shape[-1])], 
            initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [ int(shape[-1])], 
            initializer=tf.constant_initializer(1), trainable=trainable)
        
        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)
        
        x = (x - mean) /  tf.sqrt(variance + epsilon)

        return gamma * x + beta

def byetenet_residual_block(input_, dilation, layer_no, 
    residual_channels, filter_width,
    causal = True, train = True):
        block_type = "decoder" if causal else "encoder"
        block_name = "bytenet_{}_layer_{}_{}".format(block_type, layer_no, dilation)
        with tf.variable_scope(block_name):
            input_ln = layer_normalization(input_, name="ln1", trainable = train)
            relu1 = tf.nn.relu(input_ln)
            conv1 = conv1d(relu1, residual_channels, name = "conv1d_1")
            conv1 = layer_normalization(conv1, name="ln2", trainable = train)
            relu2 = tf.nn.relu(conv1)
            
            dilated_conv = conv1d(relu2, residual_channels, 
                dilation, filter_width,
                causal = causal,
                name = "dilated_conv"
                )
            print dilated_conv
            dilated_conv = layer_normalization(dilated_conv, name="ln3", trainable = train)
            relu3 = tf.nn.relu(dilated_conv)
            conv2 = conv1d(relu3, 2 * residual_channels, name = 'conv1d_2')
            return input_ + conv2

def init_weight(dim_in, dim_out, name=None, stddev=1.0):
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

def init_bias(dim_out, name=None):
    return tf.Variable(tf.zeros([dim_out]), name=name)