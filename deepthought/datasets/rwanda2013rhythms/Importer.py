"""
new importer to use with MultiChannelEEGDataset
"""
__author__ = 'sstober'

import os
import glob
import csv
import math

import logging
log = logging.getLogger(__name__)

import numpy as np
import theano

from pylearn2.utils.timing import log_timing
from deepthought.util.fs_util import save, load
from deepthought.datasets.rwanda2013rhythms import LabelConverter


def load_data_file(filename):

    #data = np.loadtxt(filename, dtype=float, delimiter=' ', skiprows=1)  #, autostrip=True, names=False)
    with log_timing(log, 'loading data from {}'.format(filename)):
        data = np.genfromtxt(filename,  dtype=theano.config.floatX, delimiter=' ', skip_header=1, autostrip=True)
    log.info('loaded {}'.format(data.shape))

#     print data.shape
#     print data[0]
#     print data[-1]

    return data

def load_xlsx_meta_file(filename):
    import xlrd

    book = xlrd.open_workbook(filename, encoding_override="cp1252")
    sheet = book.sheet_by_index(0)

    onsets = []
    for i in range(1, sheet.nrows):
        onsets.append([sheet.cell(i,2).value, sheet.cell(i,0).value.encode('ascii')])
        log.debug(onsets[-1])

    return onsets

def generate_filepath_from_metadata(metadata):
    return '{}/{}.pklz'.format(
                     metadata['subject'],
                     metadata['label'],
                     )

def split_session(sourcefile_path, trial_len):

    log.info('processing {}'.format(sourcefile_path))

    datafile = glob.glob(os.path.join(sourcefile_path,'*.txt'))[0]
    metafile = glob.glob(os.path.join(sourcefile_path,'*_Trials_Onsets.xlsx'))[0]

    log.debug('data file: {}'.format(datafile))
    log.debug('meta file: {}'.format(metafile))

    onsets = load_xlsx_meta_file(metafile)
    data = load_data_file(datafile)
    log.debug(onsets)

    onsets.append([len(data), 'end']) # artificial last marker

    trials = {}
    for i in xrange(len(onsets) - 1):
        onset, label = onsets[i]
        next_onset = onsets[i+1][0]

        # rounding to integers
        onset = int(math.floor(float(onset)))
        next_onset = int(math.floor(float(next_onset)))

        next_onset = min(onset+trial_len, next_onset)

        log.debug('[{}..{}) -> {}'.format(onset, next_onset, label))
        trial_data = np.vstack(data[onset:next_onset])
        log.debug('{} samples extracted'.format(trial_data.shape))

        trials[label] = trial_data

    # filename = os.path.join(path, 'trials.pklz')
    # with log_timing(log, 'saving to {}'.format(filename)):
    #     save(filename, trials)

    return trials

def import_dataset(source_path, target_path):

#     config = load_config(default_config='../train_sda.cfg');

    # DATA_ROOT = source_path

    # DATA_ROOT = config.eeg.get('dataset_root', './')
    SAMPLE_RATE = 400 # in Hz
    TRIAL_LENGTH = 32 # in sec

    TRIAL_LENGTH += 4 # add 4s after end of presentation

    TRIAL_SAMPLE_LENGTH = SAMPLE_RATE * TRIAL_LENGTH

    log.info('using dataset at {}'.format(source_path))

    '''
    Note from Dan:
    All subjects should have channels 15, 16, 17 and 18 removed [...]
    If you want to make them truly identical, you could remove channel 19 from
    the subjects with more channels, although this should be 'good' data.
    '''
    bad_channels = {}
    bad_channels[1]  = [5, 6,                   15, 16, 17, 18,  20, 21]
    bad_channels[2]  = [      7, 8,             15, 16, 17, 18,  20, 21]
    bad_channels[3]  = [5, 6,                   15, 16, 17, 18,  20, 21]
    bad_channels[4]  = [      7, 8,             15, 16, 17, 18,  20, 21]
    bad_channels[5]  = [      7, 8,             15, 16, 17, 18,  20, 21]
    bad_channels[6]  = [      7, 8, 9,  12,     15, 16, 17, 18         ]
    bad_channels[7]  = [5, 6,           12,     15, 16, 17, 18,  20    ]
    bad_channels[8]  = [      7, 8,             15, 16, 17, 18,  20, 21]
    bad_channels[9]  = [5, 6,           12,     15, 16, 17, 18,  20    ]
    bad_channels[10] = [5, 6,                   15, 16, 17, 18,  20, 21]
    bad_channels[11] = [5, 6,                   15, 16, 17, 18,  20, 21]
    bad_channels[12] = [5, 6,                   15, 16, 17, 18,  20, 21]
    bad_channels[13] = [5, 6,           12,     15, 16, 17, 18,  20    ]

    label_converter = LabelConverter()

    metadb_file = os.path.join(target_path, 'metadata_db.pklz')
    metadb = {}   # empty DB

    with log_timing(log, 'generating datasets'):
        for subject_id in xrange(1,14):
            search_path = os.path.join(source_path, 'Sub{0:03d}*'.format(subject_id))
            sourcefile_path = glob.glob(search_path)

            if sourcefile_path is None or len(sourcefile_path) == 0:
                log.warn('nothing found at {}'.format(search_path))
                continue
            else:
                sourcefile_path = sourcefile_path[0]

            trials = split_session(sourcefile_path, TRIAL_SAMPLE_LENGTH)

            for stimulus, trial_data in trials.iteritems():
                stimulus_id = label_converter.get_stimulus_id(stimulus)
                log.debug('processing {} with {} samples and stimulus_id {}'.
                          format(stimulus,trial_data.shape,stimulus_id))

                channels = trial_data.transpose()
                trial_data = []
                channel_ids = []
                for i, channel in enumerate(channels):
                    channel_id = i+1
                    # filter bad channels
                    if channel_id in bad_channels[subject_id]:
                        log.debug('skipping bad channel {}'.format(channel_id))
                        continue

                    # convert to float32
                    channel = np.asfarray(channel, dtype='float32')

                    trial_data.append(channel)
                    channel_ids.append(channel_id)

                trial_data = np.vstack(trial_data).transpose() # fromat: (samples, channels)
                log.debug('extracted {} from channels: {}'.format(trial_data.shape, channel_ids))

                label = label_converter.get_label(stimulus_id, 'rhythm') # raw label, unsorted
                label = label_converter.shuffle_classes[label]           # sorted label id
                metadata = {
                    'subject'       : subject_id,
                    'label'         : label,
                    'meta_label'    : label_converter.get_label(stimulus_id, 'rhythm_meta'),
                    'stimulus'      : stimulus,
                    'stimulus_id'   : stimulus_id,
                    'rhythm_type'   : label_converter.get_label(stimulus_id, 'rhythm'),
                    'tempo'         : label_converter.get_label(stimulus_id, 'tempo'),
                    'audio_file'    : label_converter.get_label(stimulus_id, 'audio_file'),
                    'trial_no'      : 1,
                    'trial_type'    : 'perception',
                    'condition'     : 'n/a',
                    'channels'      : channel_ids,
                }

                # save data
                savepath = generate_filepath_from_metadata(metadata)
                save(os.path.join(target_path, savepath), (trial_data, metadata), mkdirs=True)

                # save metadata
                metadb[savepath] = metadata

                log.debug('imported {}={} as {}'.format(label, metadata['meta_label'], savepath))

        save(metadb_file, metadb, mkdirs=True)
    log.info('import finished')


if __name__ == '__main__':
    import deepthought
    from deepthought.util.config_util import init_logging
    init_logging()
    source_path = os.path.join(deepthought.DATA_PATH, 'rwanda2013rhythms', 'eeg')
    target_path = os.path.join(deepthought.DATA_PATH, 'rwanda2013rhythms', 'multichannel')
    import_dataset(source_path, target_path)