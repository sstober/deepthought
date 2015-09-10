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
            trial = trial[:,0,:]            # get rid of 1-axis
            trial = trial.T                 # put channels first

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


class STFTProcessor(object):
    def __init__(self,
                 window_size,
                 hop_size=None,
                 n_freq_bins=None,
                 include_phase=False):

        self.window_size = window_size
        if hop_size is None:
            hop_size = self.window_size / 4
        self.hop_size = hop_size
        self.n_freq_bins = n_freq_bins
        self.include_phase = include_phase

    def process(self, trials, metadata=None):
        # expecting trials in b01c format with tf layout for 01-axes
        assert trials.shape[2] == 1

        import librosa

        tfr_trials = []
        for trial in trials:
#             print trial.shape

            n_channels = trial.shape[-1]
            channels = []
            for c in range(n_channels):
                samples = trial[:,0,c]

                '''
                from http://theremin.ucsd.edu/~bmcfee/librosadoc/librosa.html
                >>> # Get a power spectrogram from a waveform y
                >>> S       = np.abs(librosa.stft(y)) ** 2
                >>> log_S   = librosa.logamplitude(S)
                '''

                S = librosa.core.stft(samples, n_fft=self.window_size, hop_length=self.hop_size)

                # mag = np.abs(S)        # magnitude spectrum
                mag = np.abs(S)**2       # power spectrum

                # include phase information if requested
                if self.include_phase:
                    # phase = np.unwrap(np.angle(S))
                    phase = np.angle(S)

                # Optionally: cut off high bands
                if self.n_freq_bins is not None:
                    mag = mag[0:self.n_freq_bins, :]
                    if self.include_phase:
                        phase = phase[0:self.n_freq_bins, :]

                s = mag

                # include phase information if requested
                if self.include_phase:
                    # normalize phase to [-1.1]
                    phase = phase / np.pi
                    s = np.vstack([s, phase])

                # transpose to fit pylearn2 layout
                s = np.transpose(s)
#                 print s.shape

                channels.append(s)

            tfr = np.asarray(channels)
#             print tfr.shape

            # tfr format: channels x samples x freqs
            # desired output format: samples x freqs x channels
            tfr = np.rollaxis(tfr, 0, 3)
#             print tfr.shape

            tfr_trials.append(tfr)

        tfr_trials = np.asarray(tfr_trials)
#         print tfr_trials.shape
        return tfr_trials


class TrialNormalizer(object):
    '''
    NOTE on normalization:
    It depends on the structure of a neural network and (even more)
    on the properties of data. There is no best normalization algorithm
    because if there would be one, it would be used everywhere by default...

    In theory, there is no requirement for the data to be normalized at all.
    This is a purely practical thing because in practice convergence could
    take forever if your input is spread out too much. The simplest would be
    to just normalize it by scaling your data to (-1,1) (or (0,1) depending
    on activation function), and in most cases it does work. If your
    algorithm converges well, then this is your answer. If not, there are
    too many possible problems and methods to outline here without knowing
    the actual data.
    '''

    def __init__(self, mode=None, log_amplitude=False, low_clip=None, high_clip=None):

        self.mode = mode
        self.log_amplitude = log_amplitude
        self.low_clip = low_clip
        self.high_clip = high_clip

    def process(self, trials, metadata=None):
        # modification is done inplace

        import librosa

        for trial in trials:
#             print 'norm', trial.shape
            n_channels = trial.shape[-1]

            for c in range(n_channels):
                s = trial[:,:,c]

                if self.log_amplitude:
                    s = librosa.logamplitude(s)

                ## normalize to mean 0, std 1
                if self.mode == 'mean0_std1':
                    # s = preprocessing.scale(s, axis=0);
                    mean = np.mean(s)
                    std = np.std(s)
                    s = (s - mean) / std

                ## normalize by linear transform to [0,1]
                elif self.mode == 'linear_0_1':
                    s -= np.min(s)
                    s = s / np.max(s)

                ## normalize by linear transform to max(abs)=1
                # FIXME: does not check for zero!
                elif self.mode == 'linear_maxabs_1':
                    s = s / np.max(abs(s))

                ## normalize by linear transform to [-1,1]
                elif self.mode == 'linear_-1_1':
                    s = -1 + 2 * (s - np.min(s)) / (np.max(s) - np.min(s))

                elif self.mode is not None:
                    raise ValueError(
                        'unsupported spectrum normalization mode {}'.format(
                            self.mode)
                     )

                if self.low_clip is not None:
                    s = np.maximum(s, self.low_clip)

                if self.high_clip is not None:
                    s = np.minimum(s, self.high_clip)

                trial[:,:,c] = s

        return trials


class FrequencyAmplitudeNormalizer(object):
    '''
    Scales amplitudes by frequency to compensate ~1/f factor.
    This could, e.g., be useful in combination with convolution over frequency bins.
    '''

    def __init__(self, freqs):

        self.freqs = freqs


    def process(self, trials, metadata=None):
        # modification is done inplace

        for trial in trials:
            # samples x freqs x channels
            assert trial.shape[1] == len(self.freqs)

            for i, freq in enumerate(self.freqs):
                trial[:,i,:] *= freq

        return trials