__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import os
import datetime

import numpy as np

import mne
from mne.io.edf.edf import read_raw_edf

from scipy.io import loadmat

from deepthought.util.fs_util import ensure_parent_dir_exists
from deepthought.datasets.openmiir.preprocessing.keystrokes import get_keystroke_mapping
from deepthought.datasets.openmiir.metadata import load_stimuli_metadata_map
from deepthought.datasets.openmiir.events import decode_event_id
from deepthought.datasets.openmiir.constants import STIMULUS_IDS


from deepthought.datasets.openmiir.events import *



def extract_events_from_file(data_root, subject, verbose=True):
    bdf_filepath = os.path.join(data_root, 'eeg', 'biosemi', '{}.bdf'.format(subject))
    markers_filepath = os.path.join(data_root, 'eeg', 'biosemi', '{}_EEG_Data.mat'.format(subject))

    raw = read_raw_edf(bdf_filepath, preload=True, verbose=verbose)

    log.debug(raw)
    log.debug(raw.info)

    events = extract_events_from_raw(raw, markers_filepath, subject, verbose)

    # Writing events
    output_filepath = os.path.join(data_root, 'eeg', 'imported', subject, '{}-eve.fif.gz'.format(subject))
    ensure_parent_dir_exists(output_filepath)
    mne.write_events(output_filepath, events)

def extract_events_from_raw(raw, markers_filepath, subject, verbose=True):
    # read events from raw BDF data
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)

    """
    Events are stored as 2D numpy array where the
    first column is the time instant and the
    last one is the event number.
    """

    if verbose:
        print_event_type_counts(events[:,2]) # for checking occurrences

    ## read the auxiliary marker file
    markers = loadmat(markers_filepath, chars_as_strings=True, squeeze_me=True)
    markers = markers['EEG_data']

    if verbose:
        log.debug(markers)


    sampleRate = 512.0 # NOTE: this is not exact

    if subject == 'Pilot3':
        ## version without lyrics/non-lyrics info
        id2stimuli = {
            1 : 'Chim Chim Cheree',
            2 : 'Take Me Out To The Ballgame',
            3 : 'Jingle Bells',
            4 : 'Mary Had A Little Lamb',
            21 : 'Emperor Waltz',
            22 : 'Harry Potter Theme',
            23 : 'Star Wars Theme',
            24 : 'Eine Kleine Nachtmusic',  # Note: there was a typo, musik is spelled with k
            0 :'noise',
        }
        durations = [1,1,1,1,1,1,1,1,1] # FIXME
    elif subject in ['P11','P12','P13','P14']:
        id2stimuli = {
            0 : 'noise',
            1 : 'Chim Chim Cheree with lyrics',
            2 : 'Take Me Out To The Ballgame with lyrics',
            3 : 'Jingle Bells with lyrics',
            4 : 'Mary Had A Little Lamb with lyrics',
            11 : 'Chim Chim Cheree without lyrics',
            12 : 'Take Me Out To The Ballgame without lyrics',
            13 : 'Jingle Bells without lyrics',
            14 : 'Mary Had A Little Lamb without lyrics',
            21 : 'The Emperor Waltz',
            22 : 'The Harry Potter Theme',
            23 : 'The Star Wars Theme',
            24 : 'Eine Kleine Nachtmusic',
        }
        durations = [1,1,1,1,1,1,1,1,1,1,1,1,1] # FIXME
    else:
        # id2stimuli = {
        #     1 : 'Chim Chim Cheree (lyrics)',
        #     2 : 'Take Me Out To The Ballgame (lyrics)',
        #     3 : 'Jingle Bells (lyrics)',
        #     4 : 'Mary Had A Little Lamb (lyrics)',
        #     11 : 'Chim Chim Cheree (no lyrics)',
        #     12 : 'Take Me Out To The Ballgame (no lyrics)',
        #     13 : 'Jingle Bells (no lyrics)',
        #     14 : 'Mary Had A Little Lamb (no lyrics)',
        #     21 : 'Emperor Waltz',
        #     22 : 'Harry Potter Theme',
        #     23 : 'Star Wars Theme',
        #     24 : 'Eine kleine Nachtmusik',
        #     0 : 'noise',
        # }
        id2stimuli = {
            0 : 'noise',
            1 : 'Chim Chim Cheree with lyrics',
            2 : 'Take Me Out To The Ballgame with lyrics',
            3 : 'Jingle Bells with lyrics',
            4 : 'Mary Had A Little Lamb with lyrics',
            11 : 'Chim Chim Cheree without lyrics',
            12 : 'Take Me Out To The Ballgame without lyrics',
            13 : 'Jingle Bells without lyrics',
            14 : 'Mary Had A Little Lamb without lyrics',
            21 : 'Emperor Waltz',
            22 : 'Harry Potter Theme',
            23 : 'Star Wars Theme',
            24 : 'Eine kleine Nachtmusik',
        }
        durations = [1,1,1,1,1,1,1,1,1,1,1,1,1] # FIXME

    stimuli2id = dict()
    for num, stimulus in id2stimuli.iteritems():
        stimuli2id[stimulus] = num

    id2conditions = {
        1 : 'perception',
        2 : 'cued imag',
        3 : 'imag fix cross',
        4 : 'imagination',
        }
    conditions2id = dict()
    for num, condition in id2conditions.iteritems():
        conditions2id[condition] = num


    durations = np.asarray(durations)
    durations = np.ceil(durations / 1000 * sampleRate)
    #     durationsMap = containers.Map(stimuliSet, durations)

    bdfEventMap = {
        16 : 1000,      # audio onsets
        32 : 1000,
        48 : 1000,
        128 : 1001,     # perception trial onset
        256 : 1002,     # cued imagination trial onset
        384 : 1003,     # uncued imagination trial onset
        512 : 1111      # noise "trial" onset
    }

    merged = []

    ## pass over bdf event data
    lastType = 0
    sampleOffset = np.nan
    for event in events:
        etype = event[2] # type;
        if etype in bdfEventMap:
            etype = bdfEventMap[etype]
        else:
            log.warn('unsupported type: {}'.format(etype))
            break

        if (etype == 1000) and (lastType == 1000):
            etype = 0 # can be deleted
        else:
            lastType = etype

        if (etype == 1001) and (np.isnan(sampleOffset)):
            sampleOffset = event[0] # time

        if etype > 0:
            time = event[0] # time
            merged.append([time, 0, etype])

    sampleOffset += 1 # this way, we guarantee t_label > t_event
    log.debug('sample offset: {}'.format(sampleOffset))

    # BEGIN BUGFIX: repair wrong trial labels
    if subject in ['P01', 'P04', 'P05', 'P06', 'P07', 'P09']: # FIXME
        block_order = None
        labels = id2stimuli.values()
        print labels
        for i, marker in enumerate(markers):
            etype = marker[0].strip() # get rid of whitespace first!
            if etype == 'FileOrder':
                block_order = marker[1]
                print 'block order: ', block_order
                within_block_i = 0
            elif etype == 'noise':
                continue # skipping noise
            elif etype in stimuli2id:
                real_label = labels[block_order[within_block_i]] # NOTE: [0] = noise
                print 'fixing label {} -> {}'.format(etype, real_label)
                marker[0] = real_label
                within_block_i += 1
        print markers
    # END BUGFIX: repair wrong trial labels

    ## pass over real labels from auxiliary file: convert timestamps to sample offsets
    SMARK_START_ROW = 2 # start at 3rd row
    TIMESTAMP_FORMAT = '%Y%m%dT%H%M%S%f'
    # find start time
    # startTime = datetime.datetime.strptime(markers[SMARK_START_ROW+1][1], TIMESTAMP_FORMAT)
    for i in range(SMARK_START_ROW+1, SMARK_START_ROW+3):
        if markers[i][0] != 'FileOrder':
            startTime = datetime.datetime.strptime(markers[i][1], TIMESTAMP_FORMAT)

    log.debug('start time: {}'.format(startTime))
    baseType = 0
    defaultType = 'perception'
    keystroke_keys = set()
    keystroke_mapping = get_keystroke_mapping(subject)

    for i in xrange(SMARK_START_ROW, len(markers)):

        # convert stimulus markers
        etype = markers[i][0] # .type;

        if etype == 'FileOrder':
            continue

        # convert time, compute difference and map to (approximate) sample
        if etype == 'keystroke':
            # FIXME: keystrokes don't have timestamps
            # take timestamp of previous event + 1
            time = datetime.datetime.strptime(markers[i-1][1], TIMESTAMP_FORMAT)
        else:
            time = datetime.datetime.strptime(markers[i][1], TIMESTAMP_FORMAT)

        elapsed_seconds = (time - startTime).total_seconds()
        time = np.floor(elapsed_seconds * sampleRate + sampleOffset)

        if etype == 'keystroke':
            time += 1

        if etype == 'Beginning of Block 2':
            # change default type for 2nd part
            defaultType = 'imagination'
            continue

        etype = etype.strip() # remove whitespace
        if etype in stimuli2id:
            baseType = stimuli2id[etype] * 10;

            ### BEGIN hack for fixing missing lyrics/non-lyrics info ###
            if subject == 'Pilot3':
                time1 = datetime.datetime.strptime(markers[i][1], TIMESTAMP_FORMAT)

                if not markers[i+1][1] == 'RightArrow': # FIXME
                    time2 = datetime.datetime.strptime(markers[i+1][1], TIMESTAMP_FORMAT)
                    length_s = (time2 - time1).total_seconds()

                    if baseType == 10 and length_s > 20.5:
                        baseType = baseType + 100
                    elif baseType == 20 and length_s > 15:
                        baseType = baseType + 100
                    elif baseType == 30 and length_s < 17:
                        baseType = baseType + 100
                    elif baseType == 40 and length_s > 20:
                        baseType = baseType + 100
            ### END hack for fixing missing lyrics/non-lyrics info ###

            etype = defaultType

        if etype in conditions2id:
            etype = baseType + conditions2id[etype]
        elif etype == 'keystroke':
            key = markers[i][1]
            keystroke_keys.add(key)
            etype = KEYSTROKE_BASE_ID + keystroke_mapping[key]
        else:
            log.warn('unsupported type: {}'.format(etype))
            continue
#             break

        merged.append([time, 0, etype])

    log.info('keystroke keys: {} mapping: {}'.format(keystroke_keys, keystroke_mapping))

    ## sort all events by sample time
    merged = np.asarray(merged, dtype=np.int)
    merged = merged[merged[:,0].argsort()] # sort by 1st column (time)

    if verbose:
        log.debug('merged event list before fine matching:')
        for event in merged:
            log.debug('{}'.format(event))

    ## 2nd pass: fine matching
    filtered = []
    for i, event in enumerate(merged):
        etype = event[2]
        if len(filtered) == 0 and etype < 1001:
            continue # skip events before 1st trial marker

        if etype < 1000:
            continue # skip approximate data from markers

        if (etype > 1000) and (etype < 1100):
            # this is the beginning of a trial -> get label
            label = 0
            # look ahead 2 steps
            for j in xrange(i+1, i+3): # check next 2
                if merged[j][2] < 1000:
                    label = merged[j][2]
                    if verbose:
                        log.debug('matched events {} and {} dt={}'.format(
                            merged[i], merged[j], merged[j][0] - merged[i][0]))
                    break

            if label > 0:
                event[2] = label # set new label
            else:
                log.warn('unmatched event: {}'.format(merged[i]))

        # NOTE: "noise" events (1111) will be kept

        filtered.append(event)

    del merged # no longer needed

    filtered = np.asarray(filtered, dtype=int)
    if verbose:
        log.debug('filtered events:')
        for event in filtered:
            log.debug(event)
        print_event_type_counts(filtered[:,2], decoder=get_event_string) # for checking occurrences

    return filtered


def default_beat_event_id_generator(stimulus_id, condition, cue, beat_count):
    if cue:
        cue = 0
    else:
        cue = 10
    return 100000 + stimulus_id * 1000 + condition * 100 + cue + beat_count

def simple_beat_event_id_generator(stimulus_id, condition, cue, beat_count):
    return 10000

def generate_beat_events(trial_events,                  # base events as stored in raw fif files
                         include_cue_beats=True,        # generate events for cue beats as well?
                         use_audio_onset=True,          # use the more precise audio onset marker (code 1000) if present
                         exclude_stimulus_ids=[],
                         exclude_condition_ids=[],
                         beat_event_id_generator=default_beat_event_id_generator,
                         sr=512.0,                      # sample rate, correct value important to compute event frames
                         verbose=False,
                         version=None):

    ## prepare return value
    beat_events = []

    ## get stimuli meta information
    meta = load_stimuli_metadata_map(version=version)
    beats = load_stimuli_metadata_map('beats', verbose=verbose, version=version)

    if include_cue_beats:
        cue_beats = load_stimuli_metadata_map('cue_beats')

        ## determine the number of cue beats
        num_cue_beats = dict()
        for stimulus_id in STIMULUS_IDS:
            num_cue_beats[stimulus_id] = \
                meta[stimulus_id]['beats_per_bar'] * meta[stimulus_id]['cue_bars']
        if verbose:
            print num_cue_beats


    ## helper function to add a single beat event
    def add_beat_event(etime, stimulus_id, condition, beat_count, cue=False):
        etype = beat_event_id_generator(stimulus_id, condition, cue, beat_count)
        beat_events.append([etime, 0, etype])
        if verbose:
            print beat_events[-1]

    ## helper function to add a batch of beat events
    def add_beat_events(etimes, stimulus_id, condition, cue=False):
        beats_per_bar = meta[stimulus_id]['beats_per_bar']
        for i, etime in enumerate(etimes):
            beat_count = (i % beats_per_bar) + 1
            add_beat_event(etime, stimulus_id, condition, beat_count, cue)

    for i, event in enumerate(trial_events):
        etype = event[2]
        etime = event[0]

        if verbose:
            print '{:4d} at {:8d}'.format(etype, etime)

        if etype >= 1000: # stimulus_id + condition
            continue

        stimulus_id, condition = decode_event_id(etype)

        if stimulus_id in exclude_stimulus_ids or condition in exclude_condition_ids:
            continue  # skip excluded

        trial_start = etime # default: use trial onset
        if use_audio_onset and condition < 3:
            # Note: conditions 3 and 4 have no audio cues
            next_event = trial_events[i+1]
            if next_event[2] == 1000: # only use if audio onset
                trial_start = next_event[0]

        if verbose:
            print 'Trial start at {}'.format(trial_start)

        if condition < 3: # cued
            offset = sr * meta[stimulus_id]['length_of_cue']

            if include_cue_beats:
                cue_beat_times = trial_start + np.floor(sr * cue_beats[stimulus_id])
                cue_beat_times = cue_beat_times[:num_cue_beats[stimulus_id]]  # truncate at num_cue_beats
                cue_beat_times = np.asarray(cue_beat_times, dtype=int)
                if verbose:
                    print cue_beat_times
                add_beat_events(cue_beat_times, stimulus_id, condition, cue=True)
        else:
            offset = 0 # no cue

        beat_times = trial_start + offset + np.floor(sr * beats[stimulus_id])
        beat_times = np.asarray(beat_times, dtype=int)
        if verbose:
            print beat_times[:5], '...'
        add_beat_events(beat_times, stimulus_id, condition)

    beat_events = np.asarray(beat_events, dtype=int)

    return beat_events


def merge_trial_and_audio_onsets(raw, use_audio_onsets=True, inplace=True, stim_channel='STI 014', verbose=None):
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)

    merged = list()
    last_trial_event = None
    for i, event in enumerate(events):
        etype = event[2]
        if etype < 1000 or etype == 1111: # trial or noise onset
            if use_audio_onsets and events[i+1][2] == 1000: # followed by audio onset
                onset = events[i+1][0]
                merged.append([onset, 0, etype])
                if verbose:
                    log.debug('merged {} + {} = {}'.format(event, events[i+1], merged[-1]))
            else:
                # either we are not interested in audio onsets or there is none
                merged.append(event)
                if verbose:
                    log.debug('kept {}'.format(merged[-1]))
        # audio onsets (etype == 1000) are not copied
        if etype > 1111: # other events (keystrokes)
            merged.append(event)
            if verbose:
                log.debug('kept other {}'.format(merged[-1]))

    merged = np.asarray(merged, dtype=int)

    if inplace:
        stim_id = raw.ch_names.index(stim_channel)
        raw._data[stim_id,:].fill(0)     # delete data in stim channel
        raw.add_events(merged)

    return merged


def decode_beat_event_type(etype):
    # etype = 100000 + stimulus_id * 1000 + condition * 100 + cue + beat_count

    etype = int(etype)
    etype -= 100000

    stimulus_id = etype / 1000
    condition = (etype % 1000) / 100  # hundreds
    cue = (etype % 100) / 10          # tens
    beat_count = etype % 10           # last digit

    return stimulus_id, condition, cue, beat_count


def filter_beat_events(events, stimulus_ids='any', conditions='any', beat_counts='any', cue_value='any'):
#     print 'selected stimulus ids:', stimulus_ids
#     print 'selected conditions  :', conditions
#     print 'selected beat counts :', beat_counts
    filtered = list()

    for event in events:
        etype = event[2]
        stimulus_id, condition, cue, beat_count = decode_beat_event_type(etype)

        if (stimulus_ids == 'any' or stimulus_id in stimulus_ids) and \
                (conditions == 'any' or condition in conditions) and \
                (beat_counts == 'any' or beat_count in beat_counts) and \
                (cue_value == 'any' or cue == cue_value):
            filtered.append(event)

    return np.asarray(filtered)

def decode_trial_event_type(etype):
    stimulus_id = etype / 10
    condition = etype % 10
    return stimulus_id, condition

def filter_trial_events(events, stimulus_ids='any', conditions='any'):
#     print 'selected stimulus ids:', stimulus_ids
#     print 'selected conditions  :', conditions

    filtered = list()

    for event in events:
        etype = event[2]
        if etype >= 1000:
            continue

        stimulus_id, condition = decode_trial_event_type(etype)

        if (stimulus_ids == 'any' or stimulus_id in stimulus_ids) and \
                (conditions == 'any' or condition in conditions):
            filtered.append(event)

    return np.asarray(filtered)


def add_trial_cue_offsets(trial_events, meta, raw_info, debug=False):
    sfreq = raw_info['sfreq']

    n_processed = 0
    for stim_id in STIMULUS_IDS:
        offset = int(np.floor(meta[stim_id]['length_of_cue'] * sfreq))

        for cond in [1,2]: # cued conditions
            event_id = get_event_id(stim_id, cond)
            ids = np.where(trial_events[:,2] == event_id)
            log.debug('processing {} events with id {}, offset={}'.format(len(ids[0]),event_id, offset))

            for i in ids:
                if debug:
                    log.debug('before: {}'.format(trial_events[i]))
                trial_events[i,0] += offset
                if debug:
                    log.debug('after:  {}'.format(trial_events[i]))
            n_processed += len(ids[0])

    log.info('processed {} trials.'.format(n_processed))
    return trial_events


def remove_overlapping_events(events, tmin, tmax, sfreq):
    filtered = []
    sample_len = (tmax-tmin) * sfreq
    last_end = sample_len
    for event in events:
        if event[0] > last_end:
            filtered.append(event)
            last_end = event[0] + tmin + sample_len
    filtered = np.asarray(filtered)
    print 'kept {} of {} events'.format(len(filtered), len(events))
    return filtered