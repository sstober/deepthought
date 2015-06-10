__author__ = 'sstober'

import os
from scipy import io

def export_events_to_matlab(events, output_filepath):
    pass # TODO

def export_raw_to_matlab(raw, output_filepath):

    data, time = raw[:,:]
    print data.shape
    print time.shape

    io.savemat(output_filepath, dict(data=data), oned_as='row')
    # this can be imported into eeglab, events are extra