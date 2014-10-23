'''
Created on Apr 1, 2014

@author: sstober
'''
import os;
import glob;
import csv;
import math;

import logging;
log = logging.getLogger(__name__);

import numpy as np;
import theano;

from pylearn2.utils.timing import log_timing
from deepthought.util.fs_util import save, load;
from deepthought.datasets.rwanda2013rhythms import LabelConverter

def load_data_file(filename):

    #data = np.loadtxt(filename, dtype=float, delimiter=' ', skiprows=1); #, autostrip=True, names=False) 
    with log_timing(log, 'loading data from {}'.format(filename)):
        data = np.genfromtxt(filename,  dtype=theano.config.floatX, delimiter=' ', skip_header=1, autostrip=True);    
    log.info('loaded {}'.format(data.shape));
    
#     print data.shape;
#     print data[0];
#     print data[-1];

    return data;
    
#     with open(filename, 'rb') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True);
#         csvreader.next(); # skip header
#         for row in csvreader:
#             print row;
#             break
            
def load_csv_meta_file(filename):
    # open in Universal mode
    with open(filename, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, dialect='excel', delimiter=',');
        onsets = [];
        for line in reader:
            log.debug(line);
#             onsets[line['trial wav name']] = line['onset samples'];
#             onsets[line['onset samples']] = line['trial wav name'];
            onsets.append([line['onset samples'], line['trial wav name']]);
        return onsets;
    
def load_xlsx_meta_file(filename):
    import xlrd;
    
    book = xlrd.open_workbook(filename, encoding_override="cp1252")
    sheet = book.sheet_by_index(0);
    
    onsets = [];
    for i in range(1, sheet.nrows):
        onsets.append([sheet.cell(i,2).value, sheet.cell(i,0).value.encode('ascii')]);
        log.debug(onsets[-1]);
    
    return onsets;

def split_trial(path, trial_len):
    
    log.info('processing {}'.format(path));
    
    datafile = glob.glob(os.path.join(path,'*.txt'))[0];
    metafile = glob.glob(os.path.join(path,'*_Trials_Onsets.xlsx'))[0];
    
    log.debug('data file: {}'.format(datafile));
    log.debug('meta file: {}'.format(metafile));

    onsets = load_xlsx_meta_file(metafile);    
    data = load_data_file(datafile);
    log.debug(onsets);
    
    onsets.append([len(data), 'end']); # artificial last marker

    trials = {};
    for i in xrange(len(onsets) - 1):
        onset, label = onsets[i];
        next_onset = onsets[i+1][0];
        
        # rounding to integers
        onset = int(math.floor(float(onset)));
        next_onset = int(math.floor(float(next_onset)));
        
        next_onset = min(onset+trial_len, next_onset);
        
        log.debug('[{}..{}) -> {}'.format(onset, next_onset, label));
        trial_data = np.vstack(data[onset:next_onset]);
        log.debug('{} samples extracted'.format(trial_data.shape));
        
        trials[label] = trial_data;
        
    filename = os.path.join(path, 'trials.pklz');
    with log_timing(log, 'saving to {}'.format(filename)):
        save(filename, trials);
        
    return trials;




def generate_cases(subject_id, trials, bad_channels=[]):
    '''
    3x60
    4x60
    -> 12 * 60 = 720ms
    -> 60ms overlap
    '''
    
    label_converter = LabelConverter();
    
    data = [];
    labels = [];
    channel_meta = [];
    
#     trial_meta = [];
#     trial_id = 0;

    for stimulus, trial_data in trials.iteritems():
        label = label_converter.get_stimulus_id(stimulus);
        log.debug('processing {} with {} samples and label {}'.format(stimulus,trial_data.shape,label));
        channels = trial_data.transpose();
        for i, channel in enumerate(channels):
            channel_id = i+1;
            if channel_id in bad_channels:
                log.debug('skipping bad channel {}'.format(channel_id));
                continue;
    
            # convert to float32
            channel = np.asfarray(channel, dtype='float32');
            
            data.append(channel);
            labels.append(label);
#             trial_meta.append([trial_id, stimulus]);
            channel_meta.append(i);
        
#         trial_id += 1;

    data = np.vstack(data);
    labels = np.vstack(labels);
#     trial_meta = np.vstack(trial_meta);
    channel_meta = np.vstack(channel_meta);
#     subject_meta = np.vstack(subject_meta);

    log.debug('generated {} data points and {} labels '.format(data.shape, labels.shape));

#     return data, labels, trial_meta, channel_meta;
    return data, labels, channel_meta;

# if __name__ == '__main__':
def preprocess(config):
    
#     config = load_config(default_config='../train_sda.cfg');
    
    DATA_ROOT = config.eeg.get('dataset_root', './');
    SAMPLE_RATE = 400; # in Hz
    TRIAL_LENGTH = 32; # in sec
    
    TRIAL_LENGTH += 4; # add 4s after end of presentation
    
    TRIAL_SAMPLE_LENGTH = SAMPLE_RATE * TRIAL_LENGTH;    
    
    log.info('using dataset at {}'.format(DATA_ROOT));
    
    '''
    Note from Dan:
    All subjects should have channels 15, 16, 17 and 18 removed [...]
    If you want to make them truly identical, you could remove channel 19 from
    the subjects with more channels, although this should be 'good' data.
    '''    
    bad_channels = {};
    bad_channels[1]  = [5, 6,                   15, 16, 17, 18,  20, 21];
    bad_channels[2]  = [      7, 8,             15, 16, 17, 18,  20, 21];
    bad_channels[3]  = [5, 6,                   15, 16, 17, 18,  20, 21];
    bad_channels[4]  = [      7, 8,             15, 16, 17, 18,  20, 21];
    bad_channels[5]  = [      7, 8,             15, 16, 17, 18,  20, 21];
    bad_channels[6]  = [      7, 8, 9,  12,     15, 16, 17, 18         ];
    bad_channels[7]  = [5, 6,           12,     15, 16, 17, 18,  20    ];
    bad_channels[8]  = [      7, 8,             15, 16, 17, 18,  20, 21];
    bad_channels[9]  = [5, 6,           12,     15, 16, 17, 18,  20    ];
    bad_channels[10] = [5, 6,                   15, 16, 17, 18,  20, 21];
    bad_channels[11] = [5, 6,                   15, 16, 17, 18,  20, 21];
    bad_channels[12] = [5, 6,                   15, 16, 17, 18,  20, 21];
    bad_channels[13] = [5, 6,           12,     15, 16, 17, 18,  20    ];
    
    with log_timing(log, 'generating datasets'):
        for subject_id in xrange(1,14):
            search_path = os.path.join(DATA_ROOT, 'Sub{0:03d}*'.format(subject_id));
            path = glob.glob(search_path);
            
            if path is None or len(path) == 0:
                log.warn('nothing found at {}'.format(search_path));
                continue;
            else:
                path = path[0];
            
            trials_filename = os.path.join(path, 'trials.pklz');        
            
            trials = None;        
            if not os.path.isfile(trials_filename):
                log.debug('{} not found. running split_trial()'.format(trials_filename));
                trials = split_trial(path, TRIAL_SAMPLE_LENGTH);
            else:
                with log_timing(log, 'loading data from {}'.format(trials_filename)):    
                    trials = load(trials_filename);
                    
            assert trials;
             
            dataset_filename = os.path.join(path, 'dataset_13goodchannels_plus4s.pklz');
            dataset = generate_cases(subject_id, trials, bad_channels[subject_id]); # = data, labels
            with log_timing(log, 'saving dataset to {}'.format(dataset_filename)):
                save(dataset_filename, dataset);