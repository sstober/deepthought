__author__ = 'sstober'

import numpy as np
from scipy import io

def export_events_to_matlab(events, output_filepath):
    # EEGLab event structure: type, latency, urevent
    # Event latencies are stored in units of data sample points relative to (0)
    # the beginning of the continuous data matrix (EEG.data).
    eeglab_events = [[event[2], event[0], 0] for event in events]
    eeglab_events = np.asarray(eeglab_events, dtype=int)

    # print 'saving events to', output_filepath
    io.savemat(output_filepath, dict(data=eeglab_events), oned_as='row')

def export_raw_to_matlab(raw, output_filepath):

    data, time = raw[:,:]
    print data.shape
    print time.shape

    io.savemat(output_filepath, dict(data=data), oned_as='row')
    # this can be imported into eeglab, events are extra