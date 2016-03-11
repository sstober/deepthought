__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

from mne.filter import low_pass_filter, high_pass_filter, band_pass_filter, band_stop_filter, notch_filter

class MNEFilterWrapper():

    def __init__(self, type, params):
        self.type = type
        self.params = params


    def process(self, data):
        # fix for new MNE requirements
        import numpy as np
        data = np.asarray(data, dtype=np.float64)

        if self.type == 'low-pass':
            return low_pass_filter(data, **self.params)
        elif self.type == 'high-pass':
            return high_pass_filter(data, **self.params)
        elif self.type == 'band-pass':
            return band_pass_filter(data, **self.params)
        elif self.type == 'band-stop':
            return band_stop_filter(data, **self.params)
        elif self.type == 'notch':
            return notch_filter(data, **self.params)
        else:
            raise ValueError('Unsupported filter type: {}'.format(self.type))


