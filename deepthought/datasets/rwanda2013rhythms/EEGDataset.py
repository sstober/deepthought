'''
Created on Apr 2, 2014

@author: sstober
'''
import os;
import glob;

import logging;
log = logging.getLogger(__name__);

import numpy as np;

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix;
from pylearn2.utils.timing import log_timing
from pylearn2.utils import serial;

from pylearn2.format.target_format import OneHotFormatter

import librosa;  # pip install librosa
# from sklearn import preprocessing

from deepthought.util.fs_util import load;
from deepthought.datasets.rwanda2013rhythms.LabelConverter import LabelConverter

from deepthought.util.timeseries_util import frame;

class EEGDataset(DenseDesignMatrix):
    '''
    classdocs
    '''
    class Like(object):
        """
        Helper class for lazy people to load an EEGDataset with similar parameters

        Note: This is quite a hack as __new__ should return instances of Like.
              Instead, it returns the loaded EEGDataset
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

            return EEGDataset(**params)


    def __init__(self, 
                 path, suffix='',   # required data file parameters
                 subjects='all',    # optional selector (list) or 'all'
                 start_sample = 0,
                 stop_sample  = None, # optional for selection of sub-sequences
                 frame_size = -1, 
                 hop_size   = -1,   # values > 0 will lead to windowing
                 label_mode='tempo',
                 name = '',         # optional name
                 n_fft = 0,
                 n_freq_bins = None,
                 save_matrix_path = None,
                 channels = None,
                 resample = None,
                 stimulus_id_filter = None,
                 keep_metadata = False,
                 spectrum_log_amplitude = False,
                 spectrum_normalization_mode = None,
                 include_phase = False,
                 layout = 'tf'     # 2D axes layout tf=time x features or ft= features x time
                 ):
        '''
        Constructor
        '''

        # save params
        self.params = locals().copy()
        del self.params['self']
        # print self.params
        
        self.name = name;
        
        self.include_phase = include_phase;
        self.spectrum_normalization_mode = spectrum_normalization_mode;
        self.spectrum_log_amplitude = spectrum_log_amplitude;
        
        self.datafiles = [];
        subject_paths = glob.glob(os.path.join(path, 'Sub*'));
        for path in subject_paths:
            dataset_filename = os.path.join(path, 'dataset'+suffix+'.pklz');
            if os.path.isfile(dataset_filename):   
                log.debug('addding {}'.format(dataset_filename));
                self.datafiles.append(dataset_filename);
            else:
                log.warn('file does not exists {}'.format(dataset_filename));                
        self.datafiles.sort();
        
        if subjects == 'all':
            subjects = np.arange(0,len(self.datafiles));            
        assert subjects is not None and len(subjects) > 0;
        
        self.label_mode = label_mode;
        self.label_converter = LabelConverter();
        
        if stimulus_id_filter is None:
            stimulus_id_filter = [];
        self.stimulus_id_filter = stimulus_id_filter;
        
        self.subject_partitions = []; # used to keep track of original subjects
        self.sequence_partitions = []; # used to keep track of original sequences
        self.trial_partitions = []; # keeps track of original trials
        
        # metadata: [subject, trial_no, stimulus, channel, start, ]
        self.metadata = [];
        
        sequences = [];
        labels = [];
        n_sequences = 0;
        last_raw_label = -1;
        for i in xrange(len(self.datafiles)):
            if i in subjects:
                with log_timing(log, 'loading data from {}'.format(self.datafiles[i])): 
                    self.subject_partitions.append(n_sequences);                            # save start of next subject
                    
                    subject_sequences, subject_labels, channel_meta = load(self.datafiles[i]);
                    
                    subject_trial_no = -1;
                    
                    for j in xrange(len(subject_sequences)):
                        l = subject_labels[j];                                              # get raw label

                        if l in stimulus_id_filter:
#                             log.debug('skipping stimulus {}'.format(l));
                            continue;

                        c = channel_meta[j][0];
                        
                        if channels is not None and not c in channels:                      # apply optional channel filter
                            log.debug('skipping channel {}'.format(c));
                            continue;
                        
                        self.sequence_partitions.append(n_sequences);                       # save start of next sequence                        
                        
                        if l != last_raw_label:                                             # if raw label changed...
                            self.trial_partitions.append(n_sequences);                      # ...save start of next trial
                            subject_trial_no += 1;                                          # increment subject_trial_no counter
                            last_raw_label = l;
                        
                        l = self.label_converter.get_label(l[0], self.label_mode);          # convert to label_mode view
                        
                        s = subject_sequences[j];                        
                        s = s[start_sample:stop_sample];                                    # get sub-sequence in original space

                        # down-sample if requested
                        if resample is not None and resample[0] != resample[1]:
                            s = librosa.resample(s, resample[0], resample[1]);
                                                                                                
                        if n_fft is not None and n_fft > 0:                          # Optionally:
                                                                                            #     transform to spectogram
                            hop_length = n_fft / 4;
        
                            '''
                            from http://theremin.ucsd.edu/~bmcfee/librosadoc/librosa.html
                            >>> # Get a power spectrogram from a waveform y
                            >>> S       = np.abs(librosa.stft(y)) ** 2
                            >>> log_S   = librosa.logamplitude(S)
                            '''                                     
#                             s = np.abs(librosa.core.stft(s, 
#                                                             n_fft=n_fft, 
#                                                             hop_length=hop_length)
#                                           )**2;      
                            
                            S = librosa.core.stft(s, n_fft=n_fft, hop_length=hop_length);
#                             mag = np.abs(S);        # magnitude spectrum
                            mag = np.abs(S)**2;       # power spectrum  
                            # phase = np.unwrap(np.angle(S));
                            phase = np.angle(S);
                            
                            if n_freq_bins is not None:                               # Optionally:
                                mag = mag[0:n_freq_bins, :];                          #    cut off high bands
                                phase = phase[0:n_freq_bins, :];
                                                      
                            if self.spectrum_log_amplitude:      
                                mag = librosa.logamplitude(mag);
                                
                            s = mag; # for normalization
                                                        
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
#                                 s = preprocessing.scale(s, axis=0);
                                mean = np.mean(s);
                                std = np.std(s);
                                s = (s - mean) / std;
                            
                            ## normalize by linear transform to [0,1]
                            elif self.spectrum_normalization_mode == 'linear_0_1':
                                s = s / np.max(s);
                            
                            ## normalize by linear transform to [-1,1]
                            elif self.spectrum_normalization_mode == 'linear_-1_1':
                                s = -1 + 2 * (s - np.min(s)) / (np.max(s) - np.min(s));
                                
                            elif self.spectrum_normalization_mode is not None:
                                raise ValueError(
                                                 'unsupported spectrum normalization mode {}'.format(
                                                self.spectrum_normalization_mode)
                                                 );     
                            
                            #print s.mean(axis=0)
                            #print s.std(axis=0)

                            # include phase information if requested
                            if self.include_phase:
                                # normalize phase to [-1.1]
                                phase = phase / np.pi
                                s = np.vstack([s, phase]);                                       
                            
                            # transpose to fit pylearn2 layout
                            s = np.transpose(s);
                        else:
                            # normalize to max amplitude 1
                            s = librosa.util.normalize(s);
                        
                        s = np.asfarray(s, dtype='float32');
                        
                        if frame_size > 0 and hop_size > 0:
                            s, l = self._split_sequence(s, l, frame_size, hop_size);
                        
#                         print s.shape
                        n_sequences += len(s);
                         
                        sequences.append(s);
                        labels.extend(l);
                        
                        if keep_metadata:
                            self.metadata.append({
                                        'subject'   : i,                              # subject 
                                        'trial_no'  : subject_trial_no,               # trial_no
                                        'stimulus'  : last_raw_label[0],              # stimulus 
                                        'channel'   : c,                              # channel
                                        'start'     : self.sequence_partitions[-1],   # start
                                        'stop'      : n_sequences                     # stop
                                    });
      
        # turn into numpy arrays
        sequences = np.vstack(sequences);
        print sequences.shape;
        
        labels = np.hstack(labels);        
        
        # one_hot_y = one_hot(labels)
        one_hot_formatter = OneHotFormatter(labels.max() + 1)
        one_hot_y = one_hot_formatter.format(labels)
                
        self.labels = labels; # save for later
        
        if n_fft > 0:
            sequences = np.array([sequences]);
            
            # re-arrange dimensions
            sequences = sequences.swapaxes(0,1).swapaxes(1,2).swapaxes(2,3);

            if layout == 'ft':
                sequences = sequences.swapaxes(1,2)
            
            log.debug('final dataset shape: {} (b,0,1,c)'.format(sequences.shape));
            print 'final dataset shape: {} (b,0,1,c)'.format(sequences.shape)
            super(EEGDataset, self).__init__(topo_view=sequences, y=one_hot_y, axes=['b', 0, 1, 'c']);
        else:
            # if layout == 'ft':
            #     sequences = sequences.swapaxes(1,2)

            super(EEGDataset, self).__init__(X=sequences, y=one_hot_y, axes=['b', 0, 1, 'c']);
        
        log.debug('generated dataset "{}" with shape X={} y={} labels={} '.format(self.name, self.X.shape, self.y.shape, self.labels.shape));
        
        if save_matrix_path is not None:
            matrix = DenseDesignMatrix(X=sequences, y=one_hot_y);
            with log_timing(log, 'saving DenseDesignMatrix to {}'.format(save_matrix_path)):
                serial.save(save_matrix_path, matrix);

    def get_class_labels(self):
        return self.label_converter.get_class_labels(self.label_mode);
    
    
     
    def _split_sequence(self, sequence, label, frame_length, hop_length):
#         log.debug('splitting sequence with len {} with label {} into {}-frames with hop={}'.format(
#                             len(sequence), label, frame_length, hop_length));
        labels = [];
        
        frames = frame(sequence, frame_length=frame_length, hop_length=hop_length);
    
        frame_labels = [];
        for i in range(0, frames.shape[0]):
            frame_labels.append(label);
            
        labels.append(frame_labels);
            
        return frames, labels;