__author__ = 'sstober'

import numpy as np
import theano

class WindowingProcessor(object):
    def __init__(self, window_size, hop_size=None, stack_frames=False):

        if hop_size is None:
            hop_size = window_size // 4

        self.hop_size = hop_size
        self.window_size = window_size
        self.stack_frames = stack_frames


    def process(self, trials, metadata=None):
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
#             print multi_channel_frames.shape

            if self.stack_frames:
                multi_channel_frames = np.swapaxes(multi_channel_frames, 0, 2)
#                 print multi_channel_frames.shape

            frame_trials.append(multi_channel_frames)
#             break

        frame_trials = np.vstack(frame_trials)

        # print frame_trials.shape

        return frame_trials


class CWTMorletProcessor(object):
    def __init__(self, sfreq, freqs=None, use_fft=True, n_cycles=7, zero_mean=True):
        self.sfreq = sfreq
        if freqs is None:
            freqs = np.arange(1, sfreq//2 +1, 1)
        self.freqs = freqs
        self.use_fft = use_fft
        self.n_cycles = n_cycles
        self.zero_mean = zero_mean

    def process(self, trials, metadata=None):
        # expecting trials in b01c format with tf layout for 01-axes
        assert trials.shape[2] == 1

        from mne.time_frequency import cwt_morlet

        tfr_trials = []
        for trial in trials:
#             print trial.shape
            trial = trial.squeeze()         # get rid of 1-axis
            trial = trial.T                 # put channels first
            trial = np.atleast_2d(trial)    # handle single-channel case

            tfr = cwt_morlet(trial,
                             sfreq=self.sfreq,
                             freqs=self.freqs,
                             use_fft=self.use_fft,
                             n_cycles=self.n_cycles,
                             zero_mean=self.zero_mean)

            tfr = abs(tfr) ** 2

            # tfr format: channels x freqs x samples
            # desired output format: samples x freqs x channels
            tfr = np.swapaxes(tfr, 0, 2)

#             print tfr.shape
            tfr_trials.append(tfr)

        tfr_trials = np.asarray(tfr_trials)
#         print tfr_trials.shape
        return tfr_trials