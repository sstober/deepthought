'''
Created on Jan 19, 2015

@author: sstober
'''
import os
import glob

import logging
log = logging.getLogger(__name__)

import numpy as np

from deepthought.util.fs_util import load, save

from scipy.io import loadmat

def load_eeglab_data(filepath, xlsx_metadata={}):
    
    log.info('loading EEGLab data from {}'.format(filepath))
    log.debug('metadata from xlsx: {}'.format(xlsx_metadata))
    
    # EEG, metadata = load_eeglab(filepath, dtype=np.float32)
    EEG = loadmat(filepath, chars_as_strings=True, squeeze_me=True) #, chars_as_strings=True, )
    # print EEG

    data = np.asarray(EEG['epochdata'].transpose(), dtype=np.float32)
    # print data.shape

    _metadata = EEG['metadata']
    # print _metadata

    metadata = {}

    metadata['subject'] = str(_metadata['subject'])

    metadata['raw_label'] = int(_metadata['label'])
    metadata['label'] = int(_metadata['stimulus_id'])

    metadata['stimulus'] = str(_metadata['stimulus'])
    metadata['stimulus_id'] = str(_metadata['stimulus_id'])

    metadata['trial_no'] = int(_metadata['trial_no'])
    metadata['trial_type'] = str(_metadata['trial_type'])
    metadata['condition'] = str(_metadata['condition'])

    metadata['channels'] = str(_metadata['channels']).split(',')
    metadata['sample_rate'] = int(_metadata['sample_rate'])

    log.debug('subject:     {}'.format(metadata['subject']))
    log.debug('trial_no:    {}'.format(metadata['trial_no']))
    log.debug('trial_type:  {}'.format(metadata['trial_type']))
    log.debug('condition:   {}'.format(metadata['condition']))
    log.debug('label:       {}'.format(metadata['label']))
    log.debug('stimulus:    {}'.format(metadata['stimulus']))
    log.debug('stimulus_id: {}'.format(metadata['stimulus_id']))

    log.debug('channels:    {}'.format(metadata['channels']))
    log.debug('sample rate: {} Hz'.format(metadata['sample_rate']))

    log.debug('shape:       {}'.format(data.shape))
    length = float(data.shape[0]) / metadata['sample_rate']
    log.debug('length:      {}s'.format(length))
        
    return data, metadata


def generate_filepath_from_metadata(metadata):
    return '{}/{:03d}_{:03d}.pklz'.format(
                                 # metadata['trial_type'],
                                 metadata['subject'],
                                 metadata['trial_no'],
                                 metadata['raw_label'],
                                )


def import_eeglab_sets(filepaths, target_path):
    # try load metadata-db
    metadb_file = os.path.join(target_path, 'metadata_db.pklz')
    if os.path.exists(metadb_file) and os.path.isfile(metadb_file): 
        metadb = load(metadb_file)
        log.info('metadb loaded from {}'.format(metadb_file))
    else:
        metadb = {}   # empty DB
        log.info('no metadb found at {}. using empty db'.format(metadb_file))
        
    for filepath in filepaths:
        # load extra data
        filename = os.path.basename(filepath)
        data, metadata = load_eeglab_data(filepath)
        
        # save data
        savepath = generate_filepath_from_metadata(metadata)
        save(os.path.join(target_path, savepath), (data, metadata), mkdirs=True)
        
        # save metadata
        metadb[savepath] = metadata
        save(metadb_file, metadb, mkdirs=True)
        
        log.debug('imported as {}'.format(savepath))


def import_dataset(source_path, target_path):

    # load meta infos
    # xlsx_files = glob.glob(os.path.join(source_path,'*.xlsx'))
    # xlsx_metadata = dict()
    # for xlsx_file in xlsx_files:
    #     xlsx_metadata = load_xlsx_meta_file(xlsx_file, xlsx_metadata)

    # load data
    mat_files = glob.glob(os.path.join(source_path,'*/*.mat'))

    print mat_files
    import_eeglab_sets(mat_files, target_path)