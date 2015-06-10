__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import numpy as np

KEYSTROKE_BASE_ID = 2000

def get_event_id(stimulus_id, condition):
    return stimulus_id * 10 + condition

def decode_event_id(event_id):
    if event_id < 1000:
        stimulus_id = event_id / 10
        condition = event_id % 10
        return stimulus_id, condition
    else:
        return event_id

def get_event_string(event_id):
    if event_id < 1000:
        """
        Event Ids < 1000 are trial labels
        with the last digit indicating the condition
                1 : 'perception',
                2 : 'cued imag',
                3 : 'imag fix cross',
                4 : 'imagination',
        and the remaining digits referring to the stimulus id.
        """
        stimulus_id, condition = decode_event_id(event_id)
        return 'stimulus {}, condition {}'.format(stimulus_id, condition)
    else:
        return {
            1000: 'audio onset',
            1111: 'noise',
            KEYSTROKE_BASE_ID: 'imagination failed',
            KEYSTROKE_BASE_ID+1: 'imagination okay'
        }[event_id]

def print_event_type_counts(event_types, decoder=None):
    types, counts = np.unique(event_types, return_counts=True)
    for i in xrange(len(types)):
        label = types[i]
        if decoder is not None:
            label = decoder(label)
        log.debug('{}: {}'.format(label, counts[i]))