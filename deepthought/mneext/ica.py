__author__ = 'sstober'


import logging
logger = logging.getLogger('test')
from mne.preprocessing import ICA
from mne.preprocessing.ica import _check_start_stop, _reject_data_segments
from mne.evoked import Evoked,EvokedArray
from mne.io.pick import pick_types, pick_info


class EvokedICA(ICA):
    def fit(self, inst, picks=None, start=None, stop=None, decim=None,
            reject=None, flat=None, tstep=2.0, verbose=None):
        if isinstance(inst, Evoked):
#             self._fit_evoked(inst, picks, decim, verbose)
            self._fit_evoked(inst, picks, start, stop, decim, reject, flat, tstep, verbose)
        else:
            super(EvokedICA, self).fit(inst, picks, start, stop, decim, reject, flat, tstep, verbose)
        return self

    def _fit_evoked(self, raw, picks, start, stop, decim, reject, flat, tstep, verbose):
        """Aux method
        """
        if self.current_fit != 'unfitted':
            self._reset()

        if picks is None:  # just use good data channels
            picks = pick_types(raw.info, meg=True, eeg=True, eog=False,
                               ecg=False, misc=False, stim=False,
                               ref_meg=False, exclude='bads')
        logger.info('Fitting ICA to data using %i channels. \n'
                    'Please be patient, this may take some time' % len(picks))

        if self.max_pca_components is None:
            self.max_pca_components = len(picks)
            logger.info('Inferring max_pca_components from picks.')

        self.info = pick_info(raw.info, picks)
        if self.info['comps']:
            self.info['comps'] = []
        self.ch_names = self.info['ch_names']
        start, stop = _check_start_stop(raw, start, stop)

        data = raw.data[picks, start:stop]
        print data.shape
        if decim is not None:
            data = data[:, ::decim].copy()

        if (reject is not None) or (flat is not None):
            data, self.drop_inds_ = _reject_data_segments(data, reject, flat,
                                                          decim, self.info,
                                                          tstep)

        self.n_samples_ = data.shape[1]

        data, self._pre_whitener = self._pre_whiten(data,
                                                    raw.info, picks)

        self._fit(data, self.max_pca_components, 'evoked') #'raw')

        return self