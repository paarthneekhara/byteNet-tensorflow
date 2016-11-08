import os
from os import listdir
from os.path import isfile, join
import numpy as np

SPECIAL_CHARS = {
	'eol' : 256,
	'target_init' : 257,
	'padding' : 258
}

N_SENTENCES_TRANSLATOR = 200000

def load_translation_data(source_file, target_file):
	source_lines = []
	target_lines = []
	with open(source_file) as f:
		source_lines = f.read().decode("utf-8").split('\n')
	with open(target_file) as f:
		target_lines = f.read().decode("utf-8").split('\n')

	print "SL", len(source_lines), len(target_lines)

	
	source_lines = source_lines[0:N_SENTENCES_TRANSLATOR]
	target_lines = target_lines[0:N_SENTENCES_TRANSLATOR]
	
	
	for i in range(0, len(source_lines)):
		source_lines[i] = str_to_int_list(source_lines[i])
		target_lines[i] = str_to_int_list(target_lines[i])
		# print source_lines[i], target_lines[i]
		
		if i % 1000 == 0:
			print "Loading", i
	
	
	buckets = create_buckets(source_lines, target_lines, 25)
	print "***************************"
	total = 0
	frequency_ordered_keys = []
	for key in buckets:
		frequency_ordered_keys.append((-len(buckets[key]), key ))
		
	frequency_ordered_keys.sort()
	
	return buckets, frequency_ordered_keys
	

def create_buckets(source_lines, target_lines, bucket_quant):
	buckets = {}
	for i in range(0, len(source_lines)):
		
		source_lines[i] = np.concatenate( (source_lines[i], [SPECIAL_CHARS['eol']]) )
		target_lines[i] = np.concatenate( ([SPECIAL_CHARS['target_init']], target_lines[i], [SPECIAL_CHARS['eol']]) )
		
		sl = len(source_lines[i])
		tl = len(target_lines[i])


		new_length = max(sl, tl)
		if new_length % bucket_quant > 0:
			new_length = ((new_length/25) + 1 ) * 25	
		
		s_padding = np.array( [SPECIAL_CHARS['padding'] for ctr in xrange(sl, new_length) ] )

		# NEED EXTRA PADDING FOR TRAINING.. 
		t_padding = np.array( [SPECIAL_CHARS['padding'] for ctr in xrange(tl, new_length + 1) ] )

		source_lines[i] = np.concatenate( [ source_lines[i], s_padding ] )
		target_lines[i] = np.concatenate( [ target_lines[i], t_padding ] )

		if new_length in buckets:
			buckets[new_length].append( (source_lines[i], target_lines[i]) )
		else:
			buckets[new_length] = [(source_lines[i], target_lines[i])]

		if i%1000 == 0:
			print "Loading", i
		
	return buckets

def get_batch_from_pairs(pair_list):
	source_sentences = []
	target_sentences = []
	for s, t in pair_list:
		source_sentences.append(s)
		target_sentences.append(t)

	return np.array(source_sentences), np.array(target_sentences)


def str_to_int_list(text):
	text = list(text)
	for index, item in enumerate(text):
		text[index] = ord(text[index])
	
	return np.array(text, dtype='int32')

def load_text_from_directory(dir_name):
	files = [ join(dir_name, f) for f in listdir(dir_name) if ( isfile(join(dir_name, f)) and ('.txt' in f) ) ]
	text = []
	for f in files:
		text += list(open(f).read().decode("utf-8"))
	
	for index, item in enumerate(text):
		text[index] = ord(text[index])
	
	text = np.array(text, dtype='int32')
	return text
	

def load_text_samples(dir_name, sample_size):
	text = load_text_from_directory(dir_name)
	mod_size = len(text) - len(text)%sample_size
	text = text[0:mod_size]
	text = text.reshape(-1, sample_size)
	print "MAX", np.min(text)
	return text

def main():
	load_translation_data('Data/MachineTranslation/europarl-v7.de-en.de', 'Data/MachineTranslation/europarl-v7.de-en.en')

if __name__ == '__main__':

	main()