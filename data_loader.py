import os
from os import listdir
from os.path import isfile, join
import numpy as np

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
	load_text_samples('Data', 4)

if __name__ == '__main__':
	main()