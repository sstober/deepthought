'''
Created on Apr 2, 2014

@author: sstober
'''
import os

import logging
log = logging.getLogger(__name__)

import numpy as np

from pylearn2.utils.timing import log_timing
from functools import wraps

import librosa

from deepthought.util.fs_util import load
from deepthought.util.timeseries_util import frame as compute_frames
from deepthought.datasets.eeg.channel_filter import NoChannelFilter

from deepthought.datasets.eeg.MultiChannelEEGDataset import load_datafiles_metadata
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import VectorSpace, CompositeSpace, IndexSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng

class MultiChannelEEGSequencesDataset(VectorSpacesDataset):
    """
    TODO classdocs
    """
    class Like(object):
        """
        Helper class for lazy people to load an MultiChannelEEGSequencesDataset with similar parameters

        Note: This is quite a hack as __new__ should return instances of Like.
              Instead, it returns the loaded MultiChannelEEGSequencesDataset
        """
        def __new__(Like,
                    base,             # reference to copy initialize values from
                    **override
        ):
            params = base.params.copy()
            log.debug("base params: {}".format(params))
            log.debug("override params: {}".format(override))
            for key, value in override.iteritems():
                params[key] = value
            log.debug("merged params: {}".format(params))

            return MultiChannelEEGSequencesDataset(**params)


    def __init__(self, 
                 path,
                 name = '',         # optional name
                 
                 # selectors
                 subjects='all',        # optional selector (list) or 'all'
                 trial_types='all',     # optional selector (list) or 'all'
                 trial_numbers='all',   # optional selector (list) or 'all'
                 conditions='all',      # optional selector (list) or 'all'     
                 
                 partitioner = None,            
                 
                 channel_filter = NoChannelFilter(),   # optional channel filter, default: keep all
                 channel_names = None,  # optional channel names (for metadata)
                 
                 label_map = None,      # optional conversion of labels

                 remove_dc_offset = False,  # optional subtraction of channel mean, usually done already earlier
                 resample = None,       # optional down-sampling

                 # optional sub-sequences selection
                 start_sample = 0,
                 stop_sample  = None,   # optional for selection of sub-sequences

                 # optional signal filter to by applied before spitting the signal
                 signal_filter = None,

                 # windowing parameters
                 frame_size = -1,
                 hop_size   = -1,       # values > 0 will lead to windowing
                 hop_fraction = None,   # alternative to specifying absolute hop_size
                 
                 # # optional spectrum parameters, n_fft = 0 keeps raw data
                 # n_fft = 0,
                 # n_freq_bins = None,
                 # spectrum_log_amplitude = False,
                 # spectrum_normalization_mode = None,
                 # include_phase = False,

                 flatten_channels=False,
                 # layout='tf',       # (0,1)-axes layout tf=time x features or ft=features x time

                 # save_matrix_path = None,
                 keep_metadata = False,

                 target_mode='label',
                 ):
        '''
        Constructor
        '''

        # save params
        self.params = locals().copy()
        del self.params['self']
        # print self.params
        
        # TODO: get the whole filtering into an extra class
        
        datafiles_metadata, metadb = load_datafiles_metadata(path)
        
#         print datafiles_metadata
        
        def apply_filters(filters, node):            
            if isinstance(node, dict):            
                filtered = []
                keepkeys = filters[0]
                for key, value in node.items():
                    if keepkeys == 'all' or key in keepkeys:
                        filtered.extend(apply_filters(filters[1:], value))
                return filtered
            else:
                return node # [node]
            
        
        # keep only files that match the metadata filters
        self.datafiles = apply_filters([subjects,trial_types,trial_numbers,conditions], datafiles_metadata)
        
        # copy metadata for retained files
        self.metadb = {}
        for datafile in self.datafiles:
            self.metadb[datafile] = metadb[datafile]
        
#         print self.datafiles
#         print self.metadb
        
        self.name = name

        if partitioner is not None:
            self.datafiles = partitioner.get_partition(self.name, self.metadb)
        
        # self.include_phase = include_phase
        # self.spectrum_normalization_mode = spectrum_normalization_mode
        # self.spectrum_log_amplitude = spectrum_log_amplitude

        self.sequence_partitions = [] # used to keep track of original sequences
        
        # metadata: [subject, trial_no, stimulus, channel, start, ]
        self.metadata = []
        
        sequences = []
        labels = []
        targets = []
        n_sequences = 0

        print hop_size
        if frame_size > 0 and hop_size == -1 and hop_fraction is not None:
            hop_size = np.ceil(frame_size / hop_fraction)
        print hop_size

        if target_mode == 'next':
            # get 1 more value per frame as target
            frame_size += 1
        # print 'frame size: {}'.format(frame_size)

        for i in xrange(len(self.datafiles)):        
            with log_timing(log, 'loading data from {}'.format(self.datafiles[i])): 

                # save start of next sequence
                self.sequence_partitions.append(n_sequences)

                data, metadata = load(os.path.join(path, self.datafiles[i]))
                # data, metadata = self.generate_test_data()

                label = metadata['label']
                if label_map is not None:
                    label = label_map[label]

                multi_channel_frames = []
                multi_channel_targets = []

                # process 1 channel at a time
                for channel in xrange(data.shape[1]):
                    # filter channels
                    if not channel_filter.keep_channel(channel):
                        continue

                    samples = data[:, channel]
                    # print samples

                    # subtract channel mean
                    #FIXME
                    if remove_dc_offset:
                        samples -= samples.mean()

                    # down-sample if requested
                    if resample is not None and resample[0] != resample[1]:
                        samples = librosa.resample(samples, resample[0], resample[1])

                    # apply optional signal filter after down-sampling -> requires lower order
                    if signal_filter is not None:
                        samples = signal_filter.process(samples)

                    # get sub-sequence in resampled space
                    # log.info('using samples {}..{} of {}'.format(start_sample,stop_sample, samples.shape))
                    samples = samples[start_sample:stop_sample]
                    # print start_sample, stop_sample, samples.shape

                    # if n_fft is not None and n_fft > 0: # Optionally:
                    #     ### frequency spectrum branch ###
                    #
                    #     # transform to spectogram
                    #     hop_length = n_fft / 4;
                    #
                    #     '''
                    #     from http://theremin.ucsd.edu/~bmcfee/librosadoc/librosa.html
                    #     >>> # Get a power spectrogram from a waveform y
                    #     >>> S       = np.abs(librosa.stft(y)) ** 2
                    #     >>> log_S   = librosa.logamplitude(S)
                    #     '''
                    #
                    #     S = librosa.core.stft(samples, n_fft=n_fft, hop_length=hop_length)
                    #     # mag = np.abs(S)        # magnitude spectrum
                    #     mag = np.abs(S)**2       # power spectrum
                    #
                    #     # include phase information if requested
                    #     if self.include_phase:
                    #         # phase = np.unwrap(np.angle(S))
                    #         phase = np.angle(S)
                    #
                    #     # Optionally: cut off high bands
                    #     if n_freq_bins is not None:
                    #         mag = mag[0:n_freq_bins, :]
                    #         if self.include_phase:
                    #             phase = phase[0:n_freq_bins, :]
                    #
                    #     if self.spectrum_log_amplitude:
                    #         mag = librosa.logamplitude(mag)
                    #
                    #     s = mag # for normalization
                    #
                    #     '''
                    #     NOTE on normalization:
                    #     It depends on the structure of a neural network and (even more)
                    #     on the properties of data. There is no best normalization algorithm
                    #     because if there would be one, it would be used everywhere by default...
                    #
                    #     In theory, there is no requirement for the data to be normalized at all.
                    #     This is a purely practical thing because in practice convergence could
                    #     take forever if your input is spread out too much. The simplest would be
                    #     to just normalize it by scaling your data to (-1,1) (or (0,1) depending
                    #     on activation function), and in most cases it does work. If your
                    #     algorithm converges well, then this is your answer. If not, there are
                    #     too many possible problems and methods to outline here without knowing
                    #     the actual data.
                    #     '''
                    #
                    #     ## normalize to mean 0, std 1
                    #     if self.spectrum_normalization_mode == 'mean0_std1':
                    #         # s = preprocessing.scale(s, axis=0);
                    #         mean = np.mean(s)
                    #         std = np.std(s)
                    #         s = (s - mean) / std
                    #
                    #     ## normalize by linear transform to [0,1]
                    #     elif self.spectrum_normalization_mode == 'linear_0_1':
                    #         s = s / np.max(s)
                    #
                    #     ## normalize by linear transform to [-1,1]
                    #     elif self.spectrum_normalization_mode == 'linear_-1_1':
                    #         s = -1 + 2 * (s - np.min(s)) / (np.max(s) - np.min(s))
                    #
                    #     elif self.spectrum_normalization_mode is not None:
                    #         raise ValueError(
                    #             'unsupported spectrum normalization mode {}'.format(
                    #                 self.spectrum_normalization_mode)
                    #          )
                    #
                    #     #print s.mean(axis=0)
                    #     #print s.std(axis=0)
                    #
                    #     # include phase information if requested
                    #     if self.include_phase:
                    #         # normalize phase to [-1.1]
                    #         phase = phase / np.pi
                    #         s = np.vstack([s, phase])
                    #
                    #     # transpose to fit pylearn2 layout
                    #     s = np.transpose(s)
                    #     # print s.shape
                    #
                    #     ### end of frequency spectrum branch ###
                    # else:
                        ### raw waveform branch ###

                        # normalize to max amplitude 1

                    s = librosa.util.normalize(samples)

                        # add 2nd data dimension
                        # s = s.reshape(s.shape[0], 1)
                        # print s.shape

                        ### end of raw waveform branch ###
                    
                    s = np.asfarray(s, dtype='float32')

                    if frame_size > 0 and hop_size > 0:
                        # print 'frame size: {}'.format(frame_size)
                        s = s.copy() # FIXME: THIS IS NECESSARY - OTHERWISE, THE FOLLOWING OP DOES NOT WORK!!!!
                        frames = compute_frames(s, frame_length=frame_size, hop_length=hop_size)
                        # frames = librosa.util.frame(s, frame_length=frame_size, hop_length=hop_size)
                    else:
                        frames = s
                    del s
                    # print frames.shape

                    if target_mode == 'next':
                        frame_targets = np.empty(len(frames))
                        tmp = []
                        for f, frame in enumerate(frames):
                            tmp.append(frame[:-1])
                            frame_targets[f] = frame[-1]
                        frames = np.asarray(tmp)

                        # print frames.shape
                        # for f, frm in enumerate(frames):
                        #     print frm, frame_targets[f]
                        # # FIXME: OK so far

                    if flatten_channels:
                        # add artificial channel dimension
                        frames = frames.reshape((frames.shape[0], frames.shape[1], frames.shape[2], 1))
                        # print frames.shape

                        sequences.append(frames)

                        # increment counter by new number of frames
                        n_sequences += frames.shape[0]

                        if keep_metadata:
                            # determine channel name
                            channel_name = None
                            if channel_names is not None:
                                channel_name = channel_names[channel]
                            elif 'channels' in metadata:
                                channel_name = metadata['channels'][channel]

                            self.metadata.append({
                                        'subject'   : metadata['subject'],            # subject
                                        'trial_type': metadata['trial_type'],         # trial_type
                                        'trial_no'  : metadata['trial_no'],           # trial_no
                                        'condition' : metadata['condition'],          # condition
                                        'channel'   : channel,                        # channel
                                        'channel_name' : channel_name,
                                        'start'     : self.sequence_partitions[-1],   # start
                                        'stop'      : n_sequences                     # stop
                                    })

                        for _ in xrange(frames.shape[0]):
                            labels.append(label)

                        if target_mode == 'next':
                            for next in frame_targets:
                                targets.append(next)
                    else:
                        multi_channel_frames.append(frames)
                        if target_mode == 'next':
                            multi_channel_targets.append(frame_targets)

                    ### end of channel iteration ###


                    # print np.asarray(multi_channel_frames, dtype=np.int)
                    # # FIXME: OK so far


                if not flatten_channels:
                    # turn list into array
                    multi_channel_frames = np.asfarray(multi_channel_frames, dtype='float32')
                    # [channels x frames x time x freq] -> cb01
                    # [channels x frames x time x 1] -> cb0.

                    # move channel dimension to end
                    multi_channel_frames = np.rollaxis(multi_channel_frames, 0, len(multi_channel_frames.shape))
                    # print multi_channel_frames.shape
                    log.info(multi_channel_frames.shape)

                    sequences.append(multi_channel_frames)

                    # increment counter by new number of frames
                    n_sequences += multi_channel_frames.shape[0]

                    if keep_metadata:
                        self.metadata.append({
                                    'subject'   : metadata['subject'],            # subject
                                    'trial_type': metadata['trial_type'],         # trial_type
                                    'trial_no'  : metadata['trial_no'],           # trial_no
                                    'condition' : metadata['condition'],          # condition
                                    'channel'   : 'all',                          # channel
                                    'start'     : self.sequence_partitions[-1],   # start
                                    'stop'      : n_sequences                     # stop
                                })

                    for _ in xrange(multi_channel_frames.shape[0]):
                        labels.append(label)

                    if target_mode == 'next':
                        multi_channel_targets = np.asfarray(multi_channel_targets, dtype='float32')
                        targets.append(multi_channel_targets.T)

                ### end of datafile iteration ###

        # print sequences[0].shape
        # print np.asarray(sequences[0], dtype=np.int)
        # # FIXME: looks OK
      
        # turn into numpy arrays
        sequences = np.vstack(sequences)
        # sequences = np.asarray(sequences).squeeze()

        # sequences = sequences.reshape(sequences.shape[0]*sequences.shape[1], sequences.shape[2])

        print 'sequences: {}'.format(sequences.shape)
        
        labels = np.hstack(labels)
        self.labels = labels
        print 'labels: {}'.format(labels.shape)

        if target_mode == 'label':
            targets = labels.copy()

            ## copy targets to fit SequenceDataSpace(VectorSpace) structure (*, frame_size, 12)
            # targets = targets.reshape((targets.shape[0], 1))
            # targets = np.repeat(targets, frame_size, axis=1)
            # print targets.shape
            # one_hot_formatter = OneHotFormatter(max(targets.max() + 1, len(label_map)), dtype=np.int)
            # one_hot_y = one_hot_formatter.format(targets)
            # print one_hot_y.shape

            ## copy targets to fit SequenceDataSpace(IndexSpace) structure -> (*, frame_size, 1)
            targets = targets.reshape((targets.shape[0], 1))
            targets = np.repeat(targets, frame_size, axis=1)
            targets = targets.reshape((targets.shape[0], targets.shape[1], 1))
            print targets.shape

        elif target_mode == 'next':
            targets = np.concatenate(targets)
            targets = targets.reshape((targets.shape[0], 1, targets.shape[1]))
        print 'targets: {}'.format(targets.shape)

        n_channels = sequences.shape[2]
        print 'number of channels: {}'.format(n_channels)



        # if layout == 'ft': # swap axes to (batch, feature, time, channels)
        #     sequences = sequences.swapaxes(1, 2)
            
        log.debug('final dataset shape: {} (b,0,1,c)'.format(sequences.shape))

        source = ('features', 'targets')
        # space = CompositeSpace([
        #     # VectorSequenceSpace(dim=64),
        #     SequenceSpace(VectorSpace(dim=64)),
        #     VectorSpace(dim=12),
        # ])

        if target_mode == 'label':
            space = CompositeSpace([
                SequenceDataSpace(VectorSpace(dim=n_channels)),
                # SequenceDataSpace(VectorSpace(dim=12)),
                SequenceDataSpace(IndexSpace(dim=1, max_labels=12)),
                # SequenceDataSpace(IndexSpace(dim=512, max_labels=12)),
            ])
        elif target_mode == 'next':
            space = CompositeSpace([
                # does not work with VectorSpacesDataset
                # SequenceSpace(VectorSpace(dim=64)),
                # SequenceSpace(VectorSpace(dim=64))

                SequenceDataSpace(VectorSpace(dim=n_channels)),
                SequenceDataSpace(VectorSpace(dim=n_channels))
                # VectorSpace(dim=n_channels)
            ])


        # source = ('features')
        # space = SequenceSpace(VectorSpace(dim=64))

        print 'sequences: {}'.format(sequences.shape)
        print 'targets: {}'.format(targets.shape)

        # for i, seq in enumerate(sequences):
        #     print np.asarray(seq, dtype=np.int)
        #     print np.asarray(targets[i], dtype=np.int)
        #     break
        #     # FIXME: looks OK

        # SequenceDataSpace(IndexSpace(dim=1, max_labels=self._max_labels)),
        if target_mode == 'label':
            super(MultiChannelEEGSequencesDataset, self).__init__(
                # data=(sequences, one_hot_y),  # works with vectorspace-target
                data=(sequences, targets),       # works with indexspace-target
                # data=sequences,
                data_specs=(space, source)
            )
        elif target_mode == 'next':
            super(MultiChannelEEGSequencesDataset, self).__init__(
                # data=(sequences, one_hot_y),  # works with vectorspace-target
                data=(sequences, targets),       # works with indexspace-target
                # data=sequences,
                data_specs=(space, source)
            )

        # super(MultiChannelEEGSequencesDataset, self).__init__(topo_view=sequences, y=one_hot_y, axes=['b', 0, 1, 'c'])
        
        # log.info('generated dataset "{}" with shape X={}={} y={} labels={} '.
        #          format(self.name, self.X.shape, sequences.shape, self.y.shape, self.labels.shape))

        # if save_matrix_path is not None:
        #     matrix = DenseDesignMatrix(topo_view=sequences, y=one_hot_y, axes=['b', 0, 1, 'c'])
        #     with log_timing(log, 'saving DenseDesignMatrix to {}'.format(save_matrix_path)):
        #         serial.save(save_matrix_path, matrix)



    ### copied from PennTreebankSequences - otherwise monitor complains about dataset size change ###
    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):

        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    ### copied from PennTreebankSequences - otherwise monitor complains about dataset size change ###
    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode='sequential'):

        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)


    def generate_test_data(self):
        data = np.empty((50,68), dtype=np.float32)
        for c in xrange(68):
            for t in xrange(50):
                data[t,c] = 100*c + t
            print data[:10,c]
        metadata = {
            'label': 1
        }
        return data, metadata