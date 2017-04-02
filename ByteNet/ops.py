import numpy as np
import tensorflow as tf

def time_to_batch(value, dilation):
	# FOR DILATED CONVOLUTION, code adapted from tensorflow-wavenet
	with tf.name_scope('time_to_batch'):
		(batch_size, sample_size, embedding_size) = value.get_shape().as_list()
		pad_elements = dilation - 1 - (sample_size + dilation - 1) % dilation
		padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
		reshaped = tf.reshape(padded, [-1, dilation, embedding_size])
		transposed = tf.transpose(reshaped, perm=[1, 0, 2])
		return tf.reshape(transposed, [batch_size * dilation, -1, embedding_size])

def batch_to_time(value, dilation):
	with tf.name_scope('batch_to_time'):
		(batch_size, sample_size, embedding_size) = value.get_shape().as_list()
		prepared = tf.reshape(value, [dilation, -1, embedding_size])
		transposed = tf.transpose(prepared, perm=[1, 0, 2])
		return tf.reshape(transposed,
						  [int(batch_size / dilation), -1, embedding_size])

def conv1d(input_,
		   output_channels,
		   filter_width = 1,
		   stride       = 1,
		   stddev       = 0.02,
		   name         = 'conv1d'
		  ):
	with tf.variable_scope(name):
		in_dim = input_.get_shape().as_list()[-1]

		w = tf.get_variable('W', [filter_width, in_dim, output_channels],
			initializer = tf.truncated_normal_initializer(stddev=stddev))
		b = tf.get_variable('b', [output_channels], initializer=tf.constant_initializer(0.0))

		return tf.nn.conv1d(input_, w, stride = stride, padding = 'SAME') + b

def dilated_conv1d(input_, output_channels, dilation, 
	filter_width = 1, causal = False, name = 'dilated_conv'):

	if causal:
		# padding for masked convolution
		padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
	else:
		padding = [[0, 0], [(filter_width - 1) * dilation/2, (filter_width - 1) * dilation/2], [0, 0]]

	padded = tf.pad(input_, padding)
	
	if dilation > 1:
		transformed = time_to_batch(padded, dilation)
		conv = conv1d(transformed, output_channels, filter_width, name = name)
		restored = batch_to_time(conv, dilation)
	else:
		restored = conv1d(padded, output_channels, filter_width, name = name)

	# Remove excess elements at the end
	result = tf.slice(restored,[0, 0, 0],[-1, int(input_.get_shape()[1]), -1])
	
	return result
