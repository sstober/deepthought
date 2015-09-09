__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import numpy as np

import functools

# from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import Dataset
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace, IndexSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class

import librosa

from deepthought.datasets.eeg.channel_filter import NoChannelFilter
from deepthought.datasets.selection import DatasetMetaDB
import theano

# legacy support
from deepthought.datasets.datasources import Datasource as DataSet
from deepthought.datasets.datasources import SubDatasource as Subset
from deepthought.datasets.datasources import SingleFileDatasource as DataFile

class EEGEpochsDataset(Dataset):
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

                 debug=False,
                 ):
        '''
        Constructor
        '''

        # save params
        self.params = locals().copy()
        del self.params['self']
        # print self.params

        self.name = name
        self.debug = debug

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
            # trials will be in b01c format with tf layout for 01-axes
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
        # super(EEGEpochsDataset, self).__init__(topo_view=self.trials, y=self.targets, axes=['b', 0, 1, 'c'])

        self.X = self.trials.reshape(self.trials.shape[0], np.prod(self.trials.shape[1:]))
        self.y = self.targets
        log.info('generated dataset "{}" with shape X={}={} y={} targets={} '.
                 format(self.name, self.X.shape, self.trials.shape, self.y.shape, self.targets.shape))


        # determine data specs
        features_space = Conv2DSpace(
            shape=[self.trials.shape[1], self.trials.shape[2]],
            num_channels=self.trials.shape[3]
        )
        features_source = 'features'

        targets_space = VectorSpace(dim=self.targets.shape[-1])
        targets_source = 'targets'

        space_components = [features_space, targets_space]
        source_components = [features_source, targets_source]

        # additional support for subject information
        self.subjects = sorted(list(set([meta['subject'] for meta in self.metadata])))
        space_components.extend([VectorSpace(dim=1)])
        source_components.extend(['subjects'])

        space = CompositeSpace(space_components)
        source = tuple(source_components)
        self.data_specs = (space, source)


    def get_data_specs(self):
        return self.data_specs

    def get_num_examples(self):
        return len(self.trials)


    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.
        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")


    def get(self, source, indexes):
        """
        .. todo::
            WRITEME
        """
        if self.debug:
            print 'get', indexes

        if type(indexes) is slice:
            indexes = np.arange(indexes.start, indexes.stop)
        self._validate_source(source)

        rval = []
        for so in source:

            if so == 'features':
                batch = self.trials[indexes]
            elif so == 'targets':
                batch = self.targets[indexes]
            elif so == 'subjects':
                batch = [self.subjects.index(self.metadata[i]['subject']) for i in indexes]
                batch = np.atleast_2d(np.asarray(batch, dtype=int)).T
                # print batch
            else:
                pass # TODO: support more sources

            rval.append(batch)
        return tuple(rval)


    '''
    adapted from https://github.com/vdumoulin/research/blob/master/code/pylearn2/datasets/timit.py
    '''
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::
            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(len(self.trials), batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)