__author__ = 'sstober'

from deepthought.datasets.openmiir.metadata import load_stimuli_metadata, get_stimuli_version

class BPMTargetProcessor(object):

    def __init__(self, data_root=None):
        self.meta = dict()
        for version in [1,2]:
            self.meta[version] = load_stimuli_metadata(data_root=data_root, version=version)

    def process(self, target, metadata):
        subject = metadata['subject']
        stimulus_id = metadata['stimulus_id']
        version = get_stimuli_version(subject)
        bpm = self.meta[version][stimulus_id]['bpm']
        return bpm
