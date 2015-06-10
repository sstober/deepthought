__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

"""
default use of left and right arrows
"""
LR_ARROW_MAPPING = {
    u'LeftArrow' : 0,
    u'RightArrow' : 1
}

# TODO: add more key mappings later as required

"""
assigns subjects to key mappings
"""
SUBJECT_MAP = {
    'Pilot3' : LR_ARROW_MAPPING
}

DEFAULT_MAPPING = LR_ARROW_MAPPING

def get_keystroke_mapping(subject):
    if subject in SUBJECT_MAP:
        return SUBJECT_MAP[subject]
    else:
        log.warn('No key mapping defined for {}. '
                 'Using default mapping.'.format(subject))
        return DEFAULT_MAPPING