__author__ = 'sstober'

import mne
from deepthought.datasets.openmiir.preprocessing.pipeline import load_raw, load_ica
from deepthought.datasets.openmiir.preprocessing.events import generate_beat_events
from deepthought.datasets.openmiir.metadata import get_stimuli_version
from deepthought.mneext.resample import fast_resample_mne, resample_mne_events

def load_and_preprocess_raw(subject, sfreq=None, verbose=None):

    # load the imported fif data, use the audio onsets
    raw = load_raw(subject, onsets='audio', verbose=verbose,
                   interpolate_bad_channels=True,
                   reference_mastoids=True)

    # apply bandpass filter, use 4 processes to speed things up
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    raw.filter(0.5, 30, picks=eeg_picks, filter_length='10s',
               l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft',
               n_jobs=4, verbose=True)

    # extract events
    # this comprises 240 trials, 60 noise events (1111) and 60 feedback events (2000=No, 2001=Yes)
    trial_events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
    print 'trial events:', trial_events.shape

    stimuli_version = get_stimuli_version(subject)

    # generate beat events - we need them to find the downbeat times
    beat_events = generate_beat_events(trial_events,
                                       include_cue_beats=False, # IMPORTANT!!!
    #                                    beat_event_id_generator=simple_beat_event_id_generator, # -> event_id=10000
                                       exclude_stimulus_ids=[],
                                       exclude_condition_ids=[],
                                       verbose=verbose,
                                       version=stimuli_version)
    print 'beat events:', beat_events.shape

    # resample data and eventa
    if sfreq is not None:
        orig_sfreq = raw.info['sfreq']
        fast_resample_mne(raw, sfreq, res_type='sinc_fastest', preserve_events=True, verbose=False)

        # IMPORTANT: extracted events have to be resampled, too - otherwise misalignment
        beat_events = resample_mne_events(beat_events, orig_sfreq, sfreq)
        trial_events = resample_mne_events(trial_events, orig_sfreq, sfreq)

    # load ica
    ica = load_ica(subject, description='100p_64c')
    print ica
    print ica.exclude
    raw = ica.apply(raw, exclude=ica.exclude, copy=False)

    return raw, trial_events, beat_events