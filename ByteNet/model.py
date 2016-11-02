import tensorflow as tf


# def create_variables:


class Byte_net_model:
	def __init__(self, options):
		
		self.options = options
		
		self.w_source_embedding = tf.get_variable('w_source_embedding', 
			[options['n_source_chars'], options['residual_channels']],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		self.w_target_embedding = tf.get_variable('w_target_embedding', 
			[options['n_target_chars'], options['residual_channels']],
			initializer=tf.truncated_normal_initializer(stddev=0.02))


	def build_model(self):
		options = self.options
		
		input_sentence = tf.placeholder('int32', [options['batch_size'], None])
		output_sentence = tf.placeholder('int32', [options['batch_size'], None])

		input_embedding = tf.nn.embedding_lookup(self.w_source_embedding, input_sentence)
		output_embedding = tf.nn.embedding_lookup(self.w_target_embedding, output_sentence)

	def build_encoder():
		


	# 	input_one_hot = tf.one_hot(
	# 		input_sentence, 
	# 		depth = options['input_channels'], 
	# 		axis = -1,
	# 		dtype = 'float32',
	# 		name = 'input_one_hot')
		
	# 	output_one_hot = tf.one_hot(
	# 		input_sentence, 
	# 		depth = options['output_channels'], 
	# 		axis = -1,
	# 		dtype = 'float32',
	# 		name = 'output_one_hot')



	# def _encode_layer(self, layer, dilation):


	# def _encode_n_block(self, layer, layer_no, dilation):

	# def encoder(self, input_sentence):

def main():
	options = {
		'n_source_chars' : 200,
		'n_target_chars' : 300,
		'residual_channels' : 800,
		'batch_size' : 3,

	}
	bn = Byte_net_model(options)
	bn.build_model()

if __name__ == '__main__':
	main()



