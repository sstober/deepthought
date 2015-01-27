'''
Created on May 10, 2014

@author: sstober
'''

import os;
from watchdog.events import LoggingEventHandler;
import shutil;
import gzip, cPickle;

def move_file_to(file_path, target_dir):
    ensure_dir_exists(target_dir);
    filename = os.path.basename(file_path);
    target_file = os.path.join(target_dir, filename);
    shutil.move(file_path, target_file);
    return target_file

def symlink(src, dst, override=False, ignore_errors=False):
    try:
        if os.path.exists(dst) and override:
            os.remove(dst);
        os.symlink(src, dst);
    except Exception as e:
        if not ignore_errors:
            raise e;
        
def touch(path, mkdirs=True):
    ensure_parent_dir_exists(path);
    with open(path, 'a'):
        os.utime(path, None)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def ensure_parent_dir_exists(path):
    parent_dir = os.path.dirname(path);
    if parent_dir == '':
        parent_dir = '.'
    ensure_dir_exists(parent_dir);
    
def convert_to_valid_filename(filename, allow_whitespace=False):
    if allow_whitespace:
        return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    else:
        return "".join([c for c in filename if c.isalpha() or c.isdigit()]).rstrip()
    
def load(filepath):    
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return cPickle.load(f)
    elif filepath.endswith('.pkl.gz') or filepath.endswith('.pklz'):
        with gzip.open(filepath, 'rb') as f:
            return cPickle.load(f)
    else:
        raise 'File format not supported for {}'.format(filepath);


def save(filepath, data, mkdirs=True):
    if mkdirs:    
        ensure_parent_dir_exists(filepath);
    if filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            cPickle.dump(data, f)
    elif filepath.endswith('.pkl.gz') or filepath.endswith('.pklz'):
        with gzip.open(filepath, 'wb') as f:
            cPickle.dump(data, f)
    else:
        raise Exception('File format not supported for {}'.format(filepath))

class CallbackFileSytemEventHandler(LoggingEventHandler):

    """handles all the events captured."""
    
    def __init__(self, callback):
        super(CallbackFileSytemEventHandler, self).__init__();
        self.callback = callback;

    def on_moved(self, event):
        super(CallbackFileSytemEventHandler, self).on_moved(event)
        self.callback();

    def on_created(self, event):
        super(CallbackFileSytemEventHandler, self).on_created(event)
        self.callback();

    def on_deleted(self, event):
        super(CallbackFileSytemEventHandler, self).on_deleted(event)
        self.callback();

    def on_modified(self, event):
        super(CallbackFileSytemEventHandler, self).on_modified(event)
        self.callback();
