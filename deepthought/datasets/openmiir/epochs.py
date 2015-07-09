__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import numpy as np
import mne
from deepthought.datasets.openmiir.preprocessing.events \
    import filter_trial_events, generate_beat_events, filter_beat_events
from deepthought.datasets.openmiir.metadata \
    import get_stimuli_version, load_stimuli_metadata


def get_subject_beat_epochs(
        subject,
        raw,
        events=None,
        tmin=-0.2,
        tmax=0.3,
        stimulus_ids='any',  # 1..4, 11..14, 21..24
        conditions='any',    # 1..4
        beat_counts='any',   # 1..4
        cue_value='any',     # 0=cue, 1=regular
        picks=None,
        verbose=False
    ):

    version = get_stimuli_version(subject)
    sfreq = raw.info['sfreq']

    if events is None:
        trial_events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0, verbose=verbose)
    else:
        trial_events = events

    beat_events = generate_beat_events(
        trial_events,
        include_cue_beats=True,
        sr=sfreq,
        version=version,
    )
#         print beat_events.shape

    # optional filtering
    beat_events = filter_beat_events(
        beat_events,
        stimulus_ids=stimulus_ids,
        conditions=conditions,
        beat_counts=beat_counts,
        cue_value=cue_value)
#         print beat_events.shape

    if picks is None:
        # default: all EEG channels including bad/interpolated
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])

    # extract epochs from raw data
    epochs = mne.Epochs(
        raw,
        beat_events,
        event_id=None, tmin=tmin, tmax=tmax,
        preload=True, proj=False, picks=picks, verbose=None)

    return epochs


def get_beat_epochs(
        subjects,            # list
        raws_dict,
        events_dict=None,
        tmin=-0.2,
        tmax=0.3,
        stimulus_ids='any',  # 1..4, 11..14, 21..24
        conditions='any',    # 1..4
        beat_counts='any',   # 1..4
        cue_value='any',     # 0=cue, 1=regular
        picks=None,
        verbose=False
    ):

    epochs = []
    for subject in subjects:

        raw = raws_dict[subject]

        if events_dict is None:
            trial_events = None
        else:
            trial_events = events_dict[subject]

        new_epochs = get_subject_beat_epochs(
            subject,
            raw=raw, events=trial_events,
            tmin=tmin, tmax=tmax,
            stimulus_ids=stimulus_ids, conditions=conditions,
            beat_counts=beat_counts, cue_value=cue_value,
            picks=picks,
            verbose=verbose
        )

        if len(new_epochs.info['bads']) > 0:
            new_epochs.interpolate_bads() # important!!!

        new_epochs.info['bads'] = [] # reset bad channels to allow concatenation
        new_epochs.info['projs'] = []
#         new_epochs.proj = []
#         print new_epochs.info

        epochs.append(new_epochs)

    epochs = mne.epochs.concatenate_epochs(epochs)
    if verbose:
        print epochs
    return epochs



def get_trial_epochs(raw, trial_events, stim_id, condition,
                     subject=None, stimuli_version=None, meta=None,
                     include_cue=False, picks=None, debug=False):

    assert subject is None or stimuli_version is None or meta is None

    if meta is None:
        if stimuli_version is None:
            if subject is None:
                raise RuntimeError('Either meta, stimuli_version or subject has to be specified.')
            else:
                stimuli_version = get_stimuli_version(subject)
        meta = load_stimuli_metadata(version=stimuli_version)

    events = filter_trial_events(trial_events, [stim_id], [condition])
    if debug:
        log.debug('selected events: {}'.format(len(events)))
        log.debug(events)

    start = 0
    if condition in [1,2]: # cued
        if include_cue:
            stop = meta[stim_id]['length_with_cue']
        else:
            # NOTE: start > 0 does not work; need to shift event time
            offset = int(np.floor(meta[stim_id]['length_of_cue'] * raw.info['sfreq']))
            events[:,0] += offset
            stop = meta[stim_id]['length_without_cue']
    else:
        stop = meta[stim_id]['length_without_cue']

    if debug:
        log.debug('start: {}  stop: {}'.format(start, stop))
        log.debug(events)

    if picks is None:
        # default: all EEG channels including bad/interpolated
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude=[])

    epochs = mne.Epochs(raw, events, None,
                              tmin=start, tmax=stop, preload=True,
                              proj=False, picks=picks, verbose=False)

    if debug:
        log.debug(epochs)

    return epochs
