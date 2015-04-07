__author__ = 'sstober'

import os

import logging
log = logging.getLogger(__name__)

import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.timing import log_timing

from pylearn2.format.target_format import OneHotFormatter

import librosa

from deepthought.util.fs_util import load
from deepthought.datasets.eeg.channel_filter import NoChannelFilter

from deepthought.datasets.selection import DatasetMetaDB

class DataFile(object):
    def __init__(self, filepath):
        self.filepath = filepath
        with log_timing(log, 'loading data from {}'.format(filepath)):
            tmp = load(filepath)
            if len(tmp) == 2:
                self.data, self.metadata = tmp
                self.targets = None
            elif len(tmp) == 3:
                self.data, self.metadata, self.targets = tmp
            else:
                raise ValueError('got {} objects instead of 2 or 3.'.format(len(tmp)))

class EEGEpochsDataset(DenseDesignMatrix):
    """
    TODO classdocs
    """
    class Like(object):
        """
        Helper class for lazy people to load an EEGEpochsDataset with similar parameters

        Note: This is quite a hack as __new__ should return instances of Like.
              Instead, it returns the loaded EEGEpochsDataset
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

                 selectors = dict(),

                 partitioner = None,

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

                 target_processor = None, # optional processing of the targets, e.g. zero-padding

                 layout='tf',       # (0,1)-axes layout tf=time x features or ft=features x time
                 ):
        '''
        Constructor
        '''

        # save params
        self.params = locals().copy()
        del self.params['self']
        # print self.params

        self.name = name

        metadb = DatasetMetaDB(db.metadata, selectors.keys())

        if partitioner is not None:
            pass # FIXME

        selected_trial_ids = metadb.select(selectors)
        print selected_trial_ids


        trials = list()
        labels = list()
        targets = list()
        meta = list()

        for trial_i in selected_trial_ids:

            if db.targets is None:
                # get and process label
                label = db.metadata[trial_i][label_attribute]
                if label_map is not None:
                    label = label_map[label]

            processed_trial = []

            trial = db.data[trial_i]

            if np.isnan(np.sum(trial)):
                print trial_i, trial

            assert not np.isnan(np.sum(trial))

            # process 1 channel at a time
            for channel in xrange(trial.shape[0]):
                # filter channels
                if not channel_filter.keep_channel(channel):
                    continue

                samples = trial[channel, :]

                # subtract channel mean
                if remove_dc_offset:
                    samples -= samples.mean()

                # down-sample if requested
                if resample is not None and resample[0] != resample[1]:
                    samples = librosa.resample(samples, resample[0], resample[1], res_type='sinc_best')

                # apply optional signal filter after down-sampling -> requires lower order
                if signal_filter is not None:
                    samples = signal_filter.process(samples)

                # get sub-sequence in resampled space
                # log.info('using samples {}..{} of {}'.format(start_sample,stop_sample, samples.shape))
                samples = samples[start_sample:stop_sample]

                # TODO optional channel processing

                # normalize to max amplitude 1
                s = librosa.util.normalize(samples)

                # add 2nd data dimension
                s = s.reshape(s.shape[0], 1)
                # print s.shape

                s = np.asfarray(s, dtype='float32')

                processed_trial.append(s)

                ### end of channel iteration ###

            processed_trial = np.asfarray([processed_trial], dtype='float32')

            # TODO optional trial processing
            # TODO optional windowing

            # processed_trial = processed_trial.reshape((1, processed_trial.shape))
            processed_trial = np.rollaxis(processed_trial, 1, 4)

            trials.append(processed_trial)
            meta.append(db.metadata[trial_i])

            if db.targets is None:
                labels.append(label)
            else:
                target = db.targets[trial_i]

                if np.isnan(np.sum(target)):
                    print trial_i, meta[-1], target

                assert not np.isnan(np.sum(target))

                if target_processor is not None:
                    target = target_processor.process(target)

                assert not np.isnan(np.sum(target))

                targets.append(target)

        ### end of datafile iteration ###

        # turn into numpy arrays
        self.trials = np.vstack(trials)

        assert not np.isnan(np.sum(self.trials))

        if db.targets is None:
            labels = np.hstack(labels)
            one_hot_formatter = OneHotFormatter(labels.max() + 1) # FIXME!
            one_hot_y = one_hot_formatter.format(labels)
            self.targets = labels
        else:
            self.targets = np.vstack(targets)

            assert not np.isnan(np.sum(self.targets))

        self.metadata = meta

        if layout == 'ft': # swap axes to (batch, feature, time, channels)
            self.trials = self.trials.swapaxes(1, 2)

        log.debug('final dataset shape: {} (b,0,1,c)'.format(self.trials.shape))
        super(EEGEpochsDataset, self).__init__(topo_view=self.trials, y=self.targets, axes=['b', 0, 1, 'c'])

        log.info('generated dataset "{}" with shape X={}={} y={} targets={} '.
                 format(self.name, self.X.shape, self.trials.shape, self.y.shape, self.targets.shape))
