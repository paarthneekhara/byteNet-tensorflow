import tensorflow as tf
import ops

# def create_variables:


class Byte_net_model:
	def __init__(self, options):
		'''
		options
		n_source_quant : quantization channels of source text
		n_target_quant : quantization channels of target text
		residual_channels : number of channels in internal blocks
		batch_size : Batch Size
		sample_size : Text Sample Length
		encoder_filter_width : Encoder Filter Width
		decoder_filter_width : Decoder Filter Width
		encoder_dilations : Dilation Factor for decoder layers (list)
		decoder_dilations : Dilation Factor for decoder layers (list)
		'''
		self.options = options
		
		self.w_source_embedding = tf.get_variable('w_source_embedding', 
			[options['n_source_quant'], 2*options['residual_channels']],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

		self.w_target_embedding = tf.get_variable('w_target_embedding', 
			[options['n_target_quant'], 2*options['residual_channels']],
			initializer=tf.truncated_normal_initializer(stddev=0.02))



	def build_prediction_model(self):
		options = self.options
		sentence = tf.placeholder('int32', [options['batch_size'], options['sample_size']], name = 'sentence')
		
		source_sentence = tf.slice(sentence, 
			[0,0], 
			[options['batch_size'], options['sample_size'] - 1], 
			name = 'source_sentence')
		
		target_sentence = tf.slice(sentence, 
			[0,1], 
			[options['batch_size'], options['sample_size'] - 1], 
			name = 'target_sentence')
		
		source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence, name = "source_embedding")
		decoder_output = self.decoder(source_embedding)
		loss = self.loss(decoder_output, target_sentence)
		

		# target_probab = tf.nn.softmax(decoder_output, name = 'target_probab')
		# print "target_probab", target_probab
		flat_logits = tf.reshape( decoder_output, [-1, options['n_target_quant']])
		prediction = tf.argmax(flat_logits, 1)
		
		variables = tf.trainable_variables()
		
		tensors = {
			'sentence' : sentence,
			'loss' : loss,
			'prediction' : prediction,
			'variables' : variables
		}


		return tensors

	def build_generator(self, sample_size, reuse = False):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		options = self.options
		source_sentence = tf.placeholder('int32', [1, sample_size], name = 'sentence')
		source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence, name = "source_embedding")
		decoder_output = self.decoder(source_embedding)
		flat_logits = tf.reshape( decoder_output, [-1, options['n_target_quant']])
		prediction = tf.argmax(flat_logits, 1)

		tensors = {
			'source_sentence' : source_sentence,
			'prediction' : prediction
		}

		return tensors
		
	def loss(self, decoder_output, target_sentence):
		options = self.options
		target_one_hot = tf.one_hot(target_sentence, 
			depth = options['n_target_quant'],
			dtype = tf.float32)
		

		flat_logits = tf.reshape( decoder_output, [-1, options['n_target_quant']])
		flat_targets = tf.reshape( target_one_hot, [-1, options['n_target_quant']])
		loss = tf.nn.softmax_cross_entropy_with_logits(flat_logits, flat_targets, name='decoder_cross_entropy_loss')
		loss = tf.reduce_mean(loss, name = 'decoder_mean_loss')

		return loss


	def decode_layer(self, input_, dilation, layer_no):
		options = self.options
		relu1 = tf.nn.relu(input_, name = 'dec_relu1_layer{}'.format(layer_no))
		conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'dec_conv1d_1_layer{}'.format(layer_no))
		

		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = ops.dilated_conv1d(relu2, options['residual_channels'], 
			dilation, options['decoder_filter_width'],
			causal = True, 
			name = "dec_dilated_conv_laye{}".format(layer_no)
			)
		
		relu3 = tf.nn.relu(dilated_conv, name = 'dec_relu1_layer{}'.format(layer_no))
		conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'dec_conv1d_2_layer{}'.format(layer_no))
		
		return input_ + conv2

	def decoder(self, input_):
		options = self.options
		curr_input = input_
		for layer_no, dilation in enumerate(options['decoder_dilations']):
			layer_output = self.decode_layer(curr_input, dilation, layer_no)
			curr_input = layer_output


		processed_output = ops.conv1d(tf.nn.relu(layer_output), 
			options['n_target_quant'], 
			name = 'decoder_post_processing')

		return processed_output



	def encode_layer(self, input_, dilation, layer_no):
		options = self.options
		relu1 = tf.nn.relu(input_, name = 'enc_relu1_layer{}'.format(layer_no))
		conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))
		relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
		dilated_conv = ops.dilated_conv1d(relu2, options['residual_channels'], 
			dilation, options['encoder_filter_width'],
			causal = False, 
			name = "enc_dilated_conv_layer{}".format(layer_no)
			)
		relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu1_layer{}'.format(layer_no))
		conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))
		return input_ + conv2
		
	def encoder(input_):
		# HAVEN'T TESTED ENCODER
		options = self.options
		curr_input = input_
		for layer_no, dilation in enumerate(self.options['dilations']):
			layer_output = self.encode_layer(curr_input, dilation, layer_no)
			curr_input = layer_output
		return layer_output
