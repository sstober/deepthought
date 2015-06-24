__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import OneHotFormatter

import librosa

from deepthought.datasets.eeg.channel_filter import NoChannelFilter
from deepthought.datasets.selection import DatasetMetaDB
import theano

# legacy support
from deepthought.datasets.datasources import Datasource as DataSet
from deepthought.datasets.datasources import SubDatasource as Subset
from deepthought.datasets.datasources import SingleFileDatasource as DataFile

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
                 db,                # data source
                 name = '',         # optional name

                 selectors = dict(),

                 partitioner = None,

                 channel_filter = NoChannelFilter(),   # optional channel filter, default: keep all
                 channel_names = None,  # optional channel names (for metadata)

                 label_attribute = 'label', # metadata attribute to be used as label
                 label_map = None,      # optional conversion of labels
                 use_targets = True,    # use targets if provides, otherwise labels are used

                 remove_dc_offset = False,  # optional subtraction of channel mean, usually done already earlier
                 resample = None,       # optional down-sampling
                 normalize = True,      # normalize to max=1

                 # optional sub-sequences selection
                 start_sample = 0,
                 stop_sample  = None,   # optional for selection of sub-sequences
                 zero_padding = True,   # if True (default) trials that are too short will be padded with
                                        # otherwise they will rejected.

                 # optional signal filter to by applied before spitting the signal
                 signal_filter = None,

                 trial_processors = [],     # optional processing of the trials
                 target_processor = None,   # optional processing of the targets, e.g. zero-padding
                 transformers = [],         # optional transformations of the dataset

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
        log.info('selected trials: {}'.format(selected_trial_ids))

        trials = list()
        labels = list()
        targets = list()
        meta = list()

        for trial_i in selected_trial_ids:

            trial_meta = db.metadata[trial_i]

            if use_targets:
                if targets is None:
                    target = None
                else:
                    target = db.targets[trial_i]
                    assert not np.isnan(np.sum(target))

                if target_processor is not None:
                    target = target_processor.process(target, trial_meta)

                    assert not np.isnan(np.sum(target))
            else:
                # get and process label
                label = db.metadata[trial_i][label_attribute]
                if label_map is not None:
                    label = label_map[label]

            processed_trial = []

            trial = db.data[trial_i]

            if np.isnan(np.sum(trial)):
                print trial_i, trial

            assert not np.isnan(np.sum(trial))

            rejected = False # flag for trial rejection

            trial = np.atleast_2d(trial)

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

                if stop_sample is not None and stop_sample > len(samples):
                    if zero_padding:
                        tmp = np.zeros(stop_sample)
                        tmp[:len(samples)] = samples
                        samples = tmp
                    else:
                        rejected = True
                        break # stop processing this trial

                s = samples[start_sample:stop_sample]

                # TODO optional channel processing

                # normalize to max amplitude 1
                if normalize:
                    s = librosa.util.normalize(s)

                # add 2nd data dimension
                s = s.reshape(s.shape[0], 1)
                # print s.shape

                s = np.asfarray(s, dtype=theano.config.floatX)

                processed_trial.append(s)

                ### end of channel iteration ###

            if rejected:
                continue    # next trial

            processed_trial = np.asfarray([processed_trial], dtype=theano.config.floatX)

            # processed_trial = processed_trial.reshape((1, processed_trial.shape))
            processed_trial = np.rollaxis(processed_trial, 1, 4)

            # optional (external) trial processing, e.g. windowing
            # trials will be in b01c format mit tf layout for 01-axes
            for trial_processor in trial_processors:
                processed_trial = trial_processor.process(processed_trial, trial_meta)

            trials.append(processed_trial)

            for k in range(len(processed_trial)):
                meta.append(trial_meta)

                if use_targets:
                    targets.append(target)
                else:
                    labels.append(label)

        ### end of datafile iteration ###

        # turn into numpy arrays
        self.trials = np.vstack(trials)

        assert not np.isnan(np.sum(self.trials))

        # prepare targets / labels
        if use_targets:
            self.targets = np.vstack(targets)
            assert not np.isnan(np.sum(self.targets))
        else:
            labels = np.hstack(labels)
            if label_map is None:
                one_hot_formatter = OneHotFormatter(max(labels) + 1)
            else:
                one_hot_formatter = OneHotFormatter(max(label_map.values()) + 1)
            one_hot_y = one_hot_formatter.format(labels)
            self.targets = one_hot_y

        self.metadata = meta

        if layout == 'ft': # swap axes to (batch, feature, time, channels)
            self.trials = self.trials.swapaxes(1, 2)

        # transform after finalizing the data structure
        for transformer in transformers:
            self.trials, self.targets = transformer.process(self.trials, self.targets)

        self.trials = np.asarray(self.trials, dtype=theano.config.floatX)

        log.debug('final dataset shape: {} (b,0,1,c)'.format(self.trials.shape))
        super(EEGEpochsDataset, self).__init__(topo_view=self.trials, y=self.targets, axes=['b', 0, 1, 'c'])

        log.info('generated dataset "{}" with shape X={}={} y={} targets={} '.
                 format(self.name, self.X.shape, self.trials.shape, self.y.shape, self.targets.shape))
