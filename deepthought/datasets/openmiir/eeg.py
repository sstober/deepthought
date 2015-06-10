__author__ = 'sstober'

from deepthought.datasets.eeg.biosemi64 import Biosemi64Layout

LAYOUT = Biosemi64Layout()

def recording_has_mastoid_channels(subject):
    if subject in ['Pilot3','P01','P02','P03','P04','P05','P06','P07','P08']:
        return False
    else:
        return True


