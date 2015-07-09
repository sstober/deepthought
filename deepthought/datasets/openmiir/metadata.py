__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import os
import numpy as np
import xlrd
import deepthought
from deepthought.util.fs_util import ensure_parent_dir_exists

from deepthought.datasets.openmiir.constants import STIMULUS_IDS

DEFAULT_VERSION = 1

def get_stimuli_version(subject):
    if subject in ['Pilot3','P01','P04','P05','P06','P07']:
        return 1
    else:
        return 2

def get_audio_filepath(stim_id, data_root=None, version=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')

    meta = load_stimuli_metadata(data_root=data_root, version=version)

    return os.path.join(data_root, 'audio', 'full.v{}'.format(version),
                        meta[stim_id]['audio_file'])

def load_stimuli_metadata(data_root=None, version=None, verbose=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')

    xlsx_filepath = os.path.join(data_root, 'meta', 'Stimuli_Meta.v{}.xlsx'.format(version))
    book = xlrd.open_workbook(xlsx_filepath, encoding_override="cp1252")
    sheet = book.sheet_by_index(0)

    if verbose:
        log.info('Loading stimulus metadata from {}'.format(xlsx_filepath))

    meta = dict()
    for i in range(1, 13):
        stimulus_id = int(sheet.cell(i,0).value)
        meta[stimulus_id] = {
            'id' : stimulus_id,
            'label' : sheet.cell(i,1).value.encode('ascii'),
            'audio_file' : sheet.cell(i,2).value.encode('ascii'),
            'cue_file' : sheet.cell(i,2).value.encode('ascii').replace('.wav', '_cue.wav'),
            'length_with_cue' : sheet.cell(i,3).value,
            'length_of_cue' : sheet.cell(i,4).value,
            'length_without_cue' : sheet.cell(i,5).value,
            'length_of_cue_only' : sheet.cell(i,6).value,
            'cue_bpm' : int(sheet.cell(i,7).value),
            'beats_per_bar' : int(sheet.cell(i,8).value),
            'num_bars' : int(sheet.cell(i,14).value),
            'cue_bars' : int(sheet.cell(i,15).value),
            'bpm' : int(sheet.cell(i,16).value),
            'approx_bar_length' : sheet.cell(i,11).value,
        }

        if version == 2:
            meta[stimulus_id]['bpm'] = meta[stimulus_id]['cue_bpm'] # use cue bpm

    return meta


def save_beat_times(beats, stimulus_id, cue=False, data_root=None, offset=None, overwrite=False, version=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')

    if cue:
        assert offset is None   # no offset in cue files
        beats_filepath = os.path.join(data_root, 'meta',
                                      'beats.v{}'.format(version),
                                      '{}_cue_beats.txt'.format(stimulus_id))
    else:
        beats_filepath = os.path.join(data_root, 'meta',
                                      'beats.v{}'.format(version),
                                      '{}_beats.txt'.format(stimulus_id))

    if os.path.exists(beats_filepath) and not overwrite:
        log.info('Skipping existing {}'.format(beats_filepath))
    else:
        ensure_parent_dir_exists(beats_filepath)
        with open(beats_filepath, 'w') as f:
            if offset is not None:
                f.write('# offset of cue: {}\n'.format(offset))
            for beat in beats:
                f.write('{}\n'.format(beat))
        log.info('Saved beat times in {}'.format(beats_filepath))


def load_beat_times(stimulus_id, cue=False, data_root=None, verbose=None, version=None):

    if version is None:
        version = DEFAULT_VERSION

    if data_root is None:
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')

    if cue:
        beats_filepath = os.path.join(data_root, 'meta',
                                      'beats.v{}'.format(version),
                                      '{}_cue_beats.txt'.format(stimulus_id))
    else:
        beats_filepath = os.path.join(data_root, 'meta',
                                      'beats.v{}'.format(version),
                                      '{}_beats.txt'.format(stimulus_id))

    with open(beats_filepath, 'r') as f:
        lines = f.readlines()

    beats = []
    for line in lines:
        if not line.strip().startswith('#'):
            beats.append(float(line.strip()))
    beats = np.asarray(beats)

    if verbose:
        log.info('Read {} beat times from {}'.format(len(beats), beats_filepath))

    return beats


def load_stimuli_metadata_map(key=None, data_root=None, verbose=None, version=None):

    if version is None:
        version = DEFAULT_VERSION

    # handle special case for beats
    if key == 'cue_beats':
        key = 'beats'
        cue = True
    else:
        cue = False

    if key == 'beats':
        map = dict()
        for stimulus_id in STIMULUS_IDS:
            map[stimulus_id] = load_beat_times(stimulus_id,
                                               cue=cue,
                                               data_root=data_root,
                                               verbose=None,
                                               version=version)
        return map

    meta = load_stimuli_metadata(data_root, version=version)

    if key is None:
        return meta  # return everything

    map = dict()
    for stimulus_id in STIMULUS_IDS:
        map[stimulus_id] = meta[stimulus_id][key]

    return map

