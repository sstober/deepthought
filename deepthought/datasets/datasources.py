__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

from deepthought.util.fs_util import load
from deepthought.datasets.selection import DatasetMetaDB

from pylearn2.utils.timing import log_timing

import os

class Datasource(object):
    def __init__(self, data, metadata, targets=None):
        self.data = data
        self.metadata = metadata
        self.targets = targets


class SubDatasource(object):
    def __init__(self, db, selectors):
        metadb = DatasetMetaDB(db.metadata, selectors.keys())
        selected_trial_ids = metadb.select(selectors)

        self.data = [db.data[i] for i in selected_trial_ids]
        self.metadata = [db.metadata[i] for i in selected_trial_ids]

        if hasattr(db, 'targets'):
            if db.targets is None:
                self.targets = None
            else:
                self.targets = [db.targets[i] for i in selected_trial_ids]


class SingleFileDatasource(object):
    def __init__(self, filepath):
        self.filepath = filepath
        with log_timing(log, 'loading data from {}'.format(filepath)):
            tmp = load(filepath)
            if len(tmp) == 2:
                self.data, self.metadata = tmp
                self.targets = None
            elif len(tmp) == 3:
                self.data, self.metadata, self.targets = tmp
            else:
                raise ValueError('got {} objects instead of 2 or 3.'.format(len(tmp)))


class MultiFileDatasource(object):
    def __init__(self, root_path, selectors=dict()):
        # read metadata file: dict filename -> metadata
        meta_map = load(os.path.join(root_path, 'metadata_db.pklz'))
        filenames = list(meta_map.keys())
        metadata = [meta_map[fn] for fn in filenames]

        # filter files by metadata selectors
        metadb = DatasetMetaDB(metadata, selectors.keys())
        selected_file_ids = metadb.select(selectors)
        # log.info('selected files: {}'.format(selected_file_ids))

        # load selected files
        self.data = []
        self.metadata = []
        for id in selected_file_ids:
            log.debug('loading data file #{} {}'.format(id, filenames[id]))
            f_data, f_metadata = load(os.path.join(root_path, filenames[id]))
            self.data.append(f_data)
            self.metadata.append(metadata[id])

        print len(self.data), len(self.metadata)
