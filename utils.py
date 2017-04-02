import numpy as np

def weighted_pick(weights):
	t = np.cumsum(weights)
	s = np.sum(weights)
	return int(np.searchsorted(t, np.random.rand(1)*s))

def list_to_string(ascii_list):
	return "".join(chr(a) for a in ascii_list if 0 <= a < 256)
