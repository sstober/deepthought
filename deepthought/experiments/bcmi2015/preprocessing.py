__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

from deepthought.datasets.openmiir.preprocessing.events import generate_beat_events
from deepthought.datasets.openmiir.metadata import get_stimuli_version
import deepthought.datasets.openmiir.preprocessing as preprocessing

# alias
preload = preprocessing.preload

def load_and_preprocess_raw(subject,
                            # default preprocessing params for BCMI 2015 experiments
                            onsets='audio',
                            interpolate_bad_channels=True,
                            reference_mastoids=True,
                            l_freq=0.5,
                            h_freq=30,
                            sfreq=64,
                            ica_cleaning=True,
                            verbose=None,
                            ):

    raw, trial_events = preprocessing.load_and_preprocess_raw(
        subject,
        onsets=onsets,
        interpolate_bad_channels=interpolate_bad_channels,
        reference_mastoids=reference_mastoids,
        l_freq=l_freq, h_freq=h_freq,
        sfreq=sfreq,
        ica_cleaning=ica_cleaning,
        verbose=verbose)

    stimuli_version = get_stimuli_version(subject)

    # generate beat events - we need them to find the downbeat times
    beat_events = generate_beat_events(trial_events,
                                       include_cue_beats=False, # IMPORTANT!!!
    #                                    beat_event_id_generator=simple_beat_event_id_generator, # -> event_id=10000
                                       exclude_stimulus_ids=[],
                                       exclude_condition_ids=[],
                                       verbose=verbose,
                                       version=stimuli_version)

    if verbose:
        log.debug('beat events: {}'.format(beat_events.shape))

    return raw, trial_events, beat_events


