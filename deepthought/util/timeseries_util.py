'''
Created on Apr 16, 2014

@author: sstober
'''

from numpy.lib.stride_tricks import as_strided
import numpy as np


def compute_number_of_frames(input_length, frame_length, hop_length):
    fake_input = np.zeros(input_length)
    frames = frame(fake_input, frame_length, hop_length)
    return len(frames)


def frame(y, frame_length, hop_length):
    y = np.ascontiguousarray(y) # important requirement - otherwise strides are messed up

    y_len = y.shape[0]
#     print y_len

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int( (y_len - frame_length) / hop_length)

    if len(y.shape) == 1:
        shape = (n_frames, frame_length)
        strides=(hop_length * y.itemsize, y.itemsize)
    elif len(y.shape) == 2:
        y_dim = y.shape[1]
        shape = (n_frames, frame_length, y_dim)
        strides=(hop_length * y_dim * y.itemsize, y_dim * y.itemsize, y.itemsize)
    else:
        raise ValueError('Shape of input unsupported: '+str(y.shape))
    
#     print shape
    
    y_frames = as_strided(y, shape=shape, strides=strides)
    return y_frames