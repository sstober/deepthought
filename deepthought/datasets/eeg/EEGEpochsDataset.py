__author__ = 'sstober'

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


def build_epoch_metadb(metadata):
    def tree():
        return collections.defaultdict(tree)

    def multi_dimensions(n, dtype):
        """ Creates an n-dimension dictionary where the n-th dimension is of type 'type'
        """
        if n == 0:
            return dtype()
        return collections.defaultdict(lambda:multi_dimensions(n-1, dtype))

    metadb = multi_dimensions(4, list)

#     datafiles = collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))

    for i, epoch_meta in enumerate(metadata):
        subject = epoch_meta['subject']
        trial_type = epoch_meta['trial_type']
        trial_number = epoch_meta['trial_no']
        condition = epoch_meta['condition']

        metadb[subject][trial_type][trial_number][condition].append(i)
        log.debug('{} {} {} {} : {}'.format(subject,trial_type,trial_number,condition,i))

    return metadb

class EpochsFile(object):
    def __init__(self, filepath):
        self.filepath = filepath
        with log_timing(log, 'loading data from {}'.format(filepath)):
            self.data, self.metadata = load(filepath)


class EEGEpochsDataset(DenseDesignMatrix):
    """
    TODO classdocs
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

            return EEGEpochsDataset(**params)


    def __init__(self,
                 db,
                 name = '',         # optional name

                 # selectors
                 subjects='all',        # optional selector (list) or 'all'
                 trial_types='all',     # optional selector (list) or 'all'
                 trial_numbers='all',   # optional selector (list) or 'all'
                 conditions='all',      # optional selector (list) or 'all'

                 # partitioner = None,

                 channel_filter = NoChannelFilter(),   # optional channel filter, default: keep all
                 channel_names = None,  # optional channel names (for metadata)

                 label_attribute = 'label', # metadata attribute to be used as label
                 label_map = None,      # optional conversion of labels

                 remove_dc_offset = False,  # optional subtraction of channel mean, usually done already earlier
                 resample = None,       # optional down-sampling

                 # optional sub-sequences selection
                 start_sample = 0,
                 stop_sample  = None,   # optional for selection of sub-sequences

                 # optional signal filter to by applied before spitting the signal
                 signal_filter = None,

                 # # windowing parameters
                 # frame_size = -1,
                 # hop_size   = -1,       # values > 0 will lead to windowing
                 # hop_fraction = None,   # alternative to specifying absolute hop_size

                 # # optional spectrum parameters, n_fft = 0 keeps raw data
                 # n_fft = 0,
                 # n_freq_bins = None,
                 # spectrum_log_amplitude = False,
                 # spectrum_normalization_mode = None,
                 # include_phase = False,
                 #
                 # flatten_channels=False,
                 layout='tf',       # (0,1)-axes layout tf=time x features or ft=features x time

                 # save_matrix_path = None,
                 # keep_metadata = False,
                 ):
        '''
        Constructor
        '''

        # save params
        self.params = locals().copy()
        del self.params['self']
        # print self.params

        # TODO: get the whole filtering into an extra class


        metadb = build_epoch_metadb(db.metadata)
#         print metadb

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

        if name == 'train':
            trial_numbers = list(range(1*12, 4*12))
        elif name == 'valid':
            trial_numbers = list(range(4*12, 5*12))
        elif name == 'test':
            trial_numbers = list(range(0*12, 1*12))
        else:
            raise ValueError('Unknown name: {}'.format(name))

        # keep only files that match the metadata filters
        selected_epoch_ids = apply_filters([subjects,trial_types,trial_numbers,conditions], metadb)

        # FIXME: hard-coded selector (above)
        # if partitioner is not None:
        #     self.datafiles = partitioner.get_partition(self.name, self.metadb)


        print selected_epoch_ids

#         print self.metadb

        self.name = name

        epochs = list()
        labels = list()
        meta = list()
        # self.metadata
        for epoch_i in selected_epoch_ids:

            label = db.metadata[epoch_i]['meter']  #FIXME
            if label_map is not None:
                label = label_map[label]

            processed_epoch = []

            # process 1 channel at a time
            for channel in xrange(db.data.shape[1]):
                # filter channels
                if not channel_filter.keep_channel(channel):
                    continue

                samples = db.data[epoch_i, channel, :]

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

                # normalize to max amplitude 1
                s = librosa.util.normalize(samples)

                # add 2nd data dimension
                s = s.reshape(s.shape[0], 1)
                # print s.shape

                s = np.asfarray(s, dtype='float32')

                processed_epoch.append(s)

                ### end of channel iteration ###

            processed_epoch = np.asfarray([processed_epoch], dtype='float32')

            # processed_epoch = processed_epoch.reshape((1, processed_epoch.shape))
            processed_epoch = np.rollaxis(processed_epoch, 1, 4)

            epochs.append(processed_epoch)
            labels.append(label)
            meta.append(db.metadata[epoch_i])


        ### end of datafile iteration ###

        # turn into numpy arrays
        epochs = np.vstack(epochs)
        # print sequences.shape;

        labels = np.hstack(labels)

        # one_hot_y = one_hot(labels)
        one_hot_formatter = OneHotFormatter(labels.max() + 1) # FIXME!
        one_hot_y = one_hot_formatter.format(labels)

        self.labels = labels
        self.metadata = meta

        if layout == 'ft': # swap axes to (batch, feature, time, channels)
            epochs = epochs.swapaxes(1, 2)

        log.debug('final dataset shape: {} (b,0,1,c)'.format(epochs.shape))
        super(EEGEpochsDataset, self).__init__(topo_view=epochs, y=one_hot_y, axes=['b', 0, 1, 'c'])

        log.info('generated dataset "{}" with shape X={}={} y={} labels={} '.
                 format(self.name, self.X.shape, epochs.shape, self.y.shape, self.labels.shape))
