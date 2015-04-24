__author__ = 'sstober'

import numpy as np
import theano



class WindowingProcessor(object):
    def __init__(self, window_size, hop_size=None):

        if hop_size is None:
            hop_size = window_size // 4

        self.hop_size = hop_size
        self.window_size = window_size

    def process(self, trials):
        # assuming b01c format
        # assuming tf layout of 01 dimensions, i.e. 0-axis will be windowed

        from deepthought.util.timeseries_util import frame

        frame_trials = []

        for trial in trials:
            trial = np.rollaxis(trial, -1, 0) # bring channels to front
            # print trial.shape

            multi_channel_frames = []
            for channel in trial:
                frames = frame(channel, frame_length=self.window_size, hop_length=self.hop_size)
                multi_channel_frames.append(frames)

            # turn list into array
            multi_channel_frames = np.asfarray(multi_channel_frames, dtype=theano.config.floatX)
            # [channels x frames x time x freq] -> cb01
            # [channels x frames x time x 1] -> cb0.

            # move channel dimension to end
            multi_channel_frames = np.rollaxis(multi_channel_frames, 0, 4)

            frame_trials.append(multi_channel_frames)

        frame_trials = np.vstack(frame_trials)

        # print frame_trials.shape

        return frame_trials