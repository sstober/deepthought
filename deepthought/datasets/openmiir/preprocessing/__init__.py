__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import mne
from deepthought.datasets.openmiir.preprocessing.pipeline import load_raw, load_ica
from deepthought.datasets.openmiir.preprocessing.events import generate_beat_events
from deepthought.mneext.resample import fast_resample_mne, resample_mne_events

def load_and_preprocess_raw(subject,
                            onsets='audio',
                            interpolate_bad_channels=True,
                            reference_mastoids=True,
                            l_freq=0.5,
                            h_freq=30,
                            sfreq=None,
                            ica_cleaning=True,
                            ica_name='100p_64c',
                            l_freq2=None,
                            h_freq2=None,
                            verbose=None,
                            n_jobs=4
                            ):

    # load the imported fif data, use the specified onsets
    raw = load_raw(subject,
                   onsets=onsets,
                   interpolate_bad_channels=interpolate_bad_channels,
                   reference_mastoids=reference_mastoids,
                   verbose=verbose,
                   )

    # apply bandpass filter, use 4 processes to speed things up
    log.info('Applying filter: low_cut_freq={} high_cut_freq={}'.format(l_freq, h_freq))
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=eeg_picks, filter_length='10s',
               l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft',
               n_jobs=n_jobs, verbose=verbose)

    # extract events
    # this comprises 240 trials, 60 noise events (1111) and 60 feedback events (2000=No, 2001=Yes)
    trial_events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0, verbose=verbose)
    if verbose:
        log.debug('trial events: {}'.format(trial_events.shape))

    # resample data and eventa
    if sfreq is not None:
        orig_sfreq = raw.info['sfreq']
        fast_resample_mne(raw, sfreq, res_type='sinc_fastest', preserve_events=True, verbose=False)

        # IMPORTANT: extracted events have to be resampled, too - otherwise misalignment
        trial_events = resample_mne_events(trial_events, orig_sfreq, sfreq)

    if ica_cleaning:
        # load ica
        ica = load_ica(subject, description=ica_name)
        if verbose:
            log.info('Applying ICA: {}'.format(ica))
        log.info('Excluding ICA components: {}'.format(ica.exclude))
        raw = ica.apply(raw, exclude=ica.exclude)

    if l_freq2 is not None or h_freq2 is not None:
        log.info('Applying additional filter: low_cut_freq={} high_cut_freq={}'.format(l_freq2, h_freq2))
        raw.filter(l_freq=l_freq2, h_freq=h_freq2, picks=eeg_picks, filter_length='10s',
               l_trans_bandwidth=0.1, h_trans_bandwidth=0.5, method='fft',
               n_jobs=n_jobs, verbose=verbose)

    return raw, trial_events


def preload(subjects,
            raw_cache=dict(),
            events_cache=dict(),
            **vargs):

    for subject in subjects:

        raw, trial_events = load_and_preprocess_raw(subject, **vargs)

        raw_cache[subject] = raw
        events_cache[subject] = trial_events

    return raw_cache, events_cache