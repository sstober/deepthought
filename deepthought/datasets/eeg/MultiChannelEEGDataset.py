'''
Created on Apr 2, 2014

@author: sstober
'''
import os

import logging
log = logging.getLogger(__name__)

import collections
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.timing import log_timing
from pylearn2.utils import serial

from pylearn2.format.target_format import OneHotFormatter

import librosa

from deepthought.util.fs_util import load
from deepthought.util.timeseries_util import frame
from deepthought.datasets.eeg.channel_filter import NoChannelFilter

def load_datafiles_metadata(path):
    def tree():
        return collections.defaultdict(tree)

    def multi_dimensions(n, dtype):
        """ Creates an n-dimension dictionary where the n-th dimension is of type 'type'
        """  
        if n == 0:
            return dtype()
        return collections.defaultdict(lambda:multi_dimensions(n-1, dtype))
    
    datafiles = multi_dimensions(4, list)
    
#     datafiles = collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    
    metadb = load(os.path.join(path, 'metadata_db.pklz'))
    for datafile, metadata in metadb.items():
        subject = metadata['subject']
        trial_type = metadata['trial_type']
        trial_number = metadata['trial_no']
        condition = metadata['condition']
#         datafiles[subject][trial_type][trial_number][condition].append(os.path.join(path, datafile));
        datafiles[subject][trial_type][trial_number][condition].append(datafile)
        log.debug('{} {} {} {} : {}'.format(subject,trial_type,trial_number,condition,datafile))
    
    return datafiles, metadb

class MultiChannelEEGDataset(DenseDesignMatrix):
    """
    this class is deprecated
    use EEGEpochsDataset for more flexibility
    """
    class Like(object):
        """
        Helper class for lazy people to load an MultiChannelEEGDataset with similar parameters

        Note: This is quite a hack as __new__ should return instances of Like.
              Instead, it returns the loaded MultiChannelEEGDataset
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

            return MultiChannelEEGDataset(**params)


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
                 
                 # optional spectrum parameters, n_fft = 0 keeps raw data
                 n_fft = 0,
                 n_freq_bins = None,
                 spectrum_log_amplitude = False,
                 spectrum_normalization_mode = None,
                 include_phase = False,

                 flatten_channels=False,
                 layout='tf',       # (0,1)-axes layout tf=time x features or ft=features x time

                 save_matrix_path = None,
                 keep_metadata = False,
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
        
        self.include_phase = include_phase
        self.spectrum_normalization_mode = spectrum_normalization_mode
        self.spectrum_log_amplitude = spectrum_log_amplitude

        self.sequence_partitions = [] # used to keep track of original sequences
        
        # metadata: [subject, trial_no, stimulus, channel, start, ]
        self.metadata = []
        
        sequences = []
        labels = []
        n_sequences = 0

        if frame_size > 0 and hop_size == -1 and hop_fraction is not None:
            hop_size = np.ceil(frame_size / hop_fraction)

        for i in xrange(len(self.datafiles)):        
            with log_timing(log, 'loading data from {}'.format(self.datafiles[i])): 

                # save start of next sequence
                self.sequence_partitions.append(n_sequences)

                data, metadata = load(os.path.join(path, self.datafiles[i]))

                label = metadata['label']
                if label_map is not None:
                    label = label_map[label]

                multi_channel_frames = []

                # process 1 channel at a time
                for channel in xrange(data.shape[1]):
                    # filter channels
                    if not channel_filter.keep_channel(channel):
                        continue

                    samples = data[:, channel]

                    # subtract channel mean
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

                    if n_fft is not None and n_fft > 0: # Optionally:
                        ### frequency spectrum branch ###

                        # transform to spectogram
                        hop_length = n_fft / 4;
            
                        '''
                        from http://theremin.ucsd.edu/~bmcfee/librosadoc/librosa.html
                        >>> # Get a power spectrogram from a waveform y
                        >>> S       = np.abs(librosa.stft(y)) ** 2
                        >>> log_S   = librosa.logamplitude(S)
                        '''                                     
                             
                        S = librosa.core.stft(samples, n_fft=n_fft, hop_length=hop_length)
                        # mag = np.abs(S)        # magnitude spectrum
                        mag = np.abs(S)**2       # power spectrum
                        
                        # include phase information if requested
                        if self.include_phase:
                            # phase = np.unwrap(np.angle(S))
                            phase = np.angle(S)

                        # Optionally: cut off high bands
                        if n_freq_bins is not None:
                            mag = mag[0:n_freq_bins, :]
                            if self.include_phase:
                                phase = phase[0:n_freq_bins, :]
                                                  
                        if self.spectrum_log_amplitude:      
                            mag = librosa.logamplitude(mag)
                            
                        s = mag # for normalization
                                                    
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
    
                        ## normalize to mean 0, std 1
                        if self.spectrum_normalization_mode == 'mean0_std1':
                            # s = preprocessing.scale(s, axis=0);
                            mean = np.mean(s)
                            std = np.std(s)
                            s = (s - mean) / std
                        
                        ## normalize by linear transform to [0,1]
                        elif self.spectrum_normalization_mode == 'linear_0_1':
                            s = s / np.max(s)
                        
                        ## normalize by linear transform to [-1,1]
                        elif self.spectrum_normalization_mode == 'linear_-1_1':
                            s = -1 + 2 * (s - np.min(s)) / (np.max(s) - np.min(s))
                            
                        elif self.spectrum_normalization_mode is not None:
                            raise ValueError(
                                'unsupported spectrum normalization mode {}'.format(
                                    self.spectrum_normalization_mode)
                             )
                        
                        #print s.mean(axis=0)
                        #print s.std(axis=0)
    
                        # include phase information if requested
                        if self.include_phase:
                            # normalize phase to [-1.1]
                            phase = phase / np.pi
                            s = np.vstack([s, phase])
                        
                        # transpose to fit pylearn2 layout
                        s = np.transpose(s)
                        # print s.shape

                        ### end of frequency spectrum branch ###
                    else:
                        ### raw waveform branch ###

                        # normalize to max amplitude 1
                        s = librosa.util.normalize(samples)

                        # add 2nd data dimension
                        s = s.reshape(s.shape[0], 1)
                        # print s.shape

                        ### end of raw waveform branch ###

                    s = np.asfarray(s, dtype='float32')

                    if frame_size > 0 and hop_size > 0:
                        s = s.copy() # FIXME: THIS IS NECESSARY IN MultiChannelEEGSequencesDataset - OTHERWISE, THE FOLLOWING OP DOES NOT WORK!!!!
                        frames = frame(s, frame_length=frame_size, hop_length=hop_size)
                    else:
                        frames = s
                    del s
                    # print frames.shape

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
                    else:
                        multi_channel_frames.append(frames)

                    ### end of channel iteration ###


                if not flatten_channels:
                    # turn list into array
                    multi_channel_frames = np.asfarray(multi_channel_frames, dtype='float32')
                    # [channels x frames x time x freq] -> cb01
                    # [channels x frames x time x 1] -> cb0.

                    # move channel dimension to end
                    multi_channel_frames = np.rollaxis(multi_channel_frames, 0, 4)
                    # print multi_channel_frames.shape
                    # log.debug(multi_channel_frames.shape)

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

                ### end of datafile iteration ###
      
        # turn into numpy arrays
        sequences = np.vstack(sequences)
        # print sequences.shape;
        
        labels = np.hstack(labels)
        
        # one_hot_y = one_hot(labels)
        one_hot_formatter = OneHotFormatter(labels.max() + 1) # FIXME!
        one_hot_y = one_hot_formatter.format(labels)
                
        self.labels = labels

        if layout == 'ft': # swap axes to (batch, feature, time, channels)
            sequences = sequences.swapaxes(1, 2)
            
        log.debug('final dataset shape: {} (b,0,1,c)'.format(sequences.shape))
        super(MultiChannelEEGDataset, self).__init__(topo_view=sequences, y=one_hot_y, axes=['b', 0, 1, 'c'])
        
        log.info('generated dataset "{}" with shape X={}={} y={} labels={} '.
                 format(self.name, self.X.shape, sequences.shape, self.y.shape, self.labels.shape))

        if save_matrix_path is not None:
            matrix = DenseDesignMatrix(topo_view=sequences, y=one_hot_y, axes=['b', 0, 1, 'c'])
            with log_timing(log, 'saving DenseDesignMatrix to {}'.format(save_matrix_path)):
                serial.save(save_matrix_path, matrix)