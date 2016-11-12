import numpy as np

def weighted_pick(weights):
	t = np.cumsum(weights)
	s = np.sum(weights)
	return(int(np.searchsorted(t, np.random.rand(1)*s)))


def list_to_string(ascii_list):
	res = u""
	for a in ascii_list:
		if a >= 0 and a < 256:
			res += unichr(a)
	return res
