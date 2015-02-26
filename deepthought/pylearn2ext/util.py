'''
Created on Apr 2, 2014

@author: sstober
'''

import logging; 
log = logging.getLogger(__name__); 

from pylearn2.train_extensions import TrainExtension;
from pylearn2.corruption import Corruptor;

import numpy as np;

import theano;

from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing
from pylearn2.space import CompositeSpace

from pylearn2.utils.string_utils import preprocess
import os
from shutil import copyfile
            
def process_dataset(model, dataset, data_specs=None, output_fn=None, batch_size=128):
    
    if data_specs is None:
        data_specs = (CompositeSpace((
                                model.get_input_space(), 
                                model.get_output_space())), 
                           ("features", "targets"));
    
    if output_fn is None:                
        with log_timing(log, 'compiling output_fn'):         
            minibatch = model.get_input_space().make_theano_batch();
            output_fn = theano.function(inputs=[minibatch], 
                                        outputs=model.fprop(minibatch));
    
    it = dataset.iterator(mode='sequential',
                          batch_size=batch_size,
                          data_specs=data_specs);
    y_pred = [];
    y_real = [];                
    output = [];
    for minibatch, target in it:
        out = output_fn(minibatch); # this hangs for convnet on Jeep2
        output.append(out);
        # print out
        # print out.shape
        y_pred.append(np.argmax(out, axis = 1));
        y_real.append(np.argmax(target, axis = 1));
    y_pred = np.hstack(y_pred);
    y_real = np.hstack(y_real);  
    output = np.vstack(output);
    
    return y_real, y_pred, output;

def aggregate_classification(seq_starts, y_real, y_pred, output):
    s_real = [];
    s_pred = [];
    s_predf = [];
    s_predfsq = [];
    for i in xrange(len(seq_starts)):        
        start = seq_starts[i];
        if i < len(seq_starts) - 1:
            stop = seq_starts[i+1];
        else:
            stop = None;
        
        s_real.append(y_real[start]);
        s_pred.append(np.argmax(np.bincount(y_pred[start:stop])));
        s_predf.append(np.argmax(np.sum(output[start:stop], axis=0)));      # sum of scores, then max
        s_predfsq.append(np.argmax(np.sum(np.log(output[start:stop]), axis=0)));   # experimental: sum of -log scores, then max
    
    s_real = np.hstack(s_real);
    s_pred = np.hstack(s_pred);  
    s_predf = np.hstack(s_predf); 
    s_predfsq = np.hstack(s_predfsq);   
    
    return s_real, s_pred, s_predf, s_predfsq;

class SaveYaml(TrainExtension):

    def __init__(self, save_dir):
        PYLEARN2_TRAIN_DIR = preprocess('${PYLEARN2_TRAIN_DIR}')
        PYLEARN2_TRAIN_BASE_NAME = preprocess('${PYLEARN2_TRAIN_BASE_NAME}')

        src = os.path.join(PYLEARN2_TRAIN_DIR, PYLEARN2_TRAIN_BASE_NAME)
        dst = os.path.join(save_dir, PYLEARN2_TRAIN_BASE_NAME)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(save_dir) and not os.path.isdir(save_dir):
            raise IOError("save path %s exists, not a directory" % save_dir)
        elif not os.access(save_dir, os.W_OK):
            raise IOError("permission error creating %s" % dst)

        with log_timing(log, 'copying yaml from {} to {}'.format(src, dst)):
            copyfile(src, dst)


class SaveEveryEpoch(TrainExtension):
    """
    A callback that saves a copy of the model every time 

    Parameters
    ----------
    save_path : str
        Path to save the model to
    """
    def __init__(self, save_path, save_prefix='cnn_epoch'):
        self.__dict__.update(locals())        

    def on_monitor(self, model, dataset, algorithm):
        
        epoch = algorithm.monitor._epochs_seen;
        model_file = self.save_path + self.save_prefix + str(epoch) + '.pkl'; 
        
        with log_timing(log, 'saving model to {}'.format(model_file)):
            serial.save(model_file, model, on_overwrite = 'backup')



class LoggingCallback(TrainExtension):
    def __init__(self, name='', obj_channels=None, obj_channel=None):
        self.name = name;
        self.obj_channels = [];
        if obj_channel is not None:
            self.obj_channels.append(obj_channel);
        if obj_channels is not None:
            self.obj_channels.extend(obj_channels);

    def on_monitor(self, model, dataset, algorithm):
                
        epoch = algorithm.monitor._epochs_seen;
        lr = algorithm.monitor.channels['learning_rate'].val_shared.get_value();
        t_epoch = algorithm.monitor.channels['training_seconds_this_epoch'].val_shared.get_value();
#         max_norms = algorithm.monitor.channels['training_seconds_this_epoch'].val_shared.get_value();
        
        log_string = 'running {} (lr={:.7f}), epoch {},'.format(
                         self.name,
                         float(lr), #algorithm.learning_rate.get_value(),
                         epoch);
                         
        for objective in self.obj_channels:
            if objective in algorithm.monitor.channels:
                v = algorithm.monitor.channels[objective].val_shared.get_value();
                log_string += '{} {:.4f},'.format(objective, float(v));
            
        log_string += 't_epoch={0}'.format(t_epoch); 
        
        log.info(log_string);


class LoggingCorruptor(Corruptor):
    '''
    decorator for an actual corruptor
    logs whenever the corruptor is called 
    '''
    
    def __init__(self, corruptor, name=''):
        self.name = name;
        self._corruptor = corruptor;
        self.corruption_level = self._corruptor.corruption_level; # copy value so that it can be read and changed
    
    def __call__(self, inputs):
        self._corruptor.corruption_level = self.corruption_level; # make sure of correct value before call
        log.debug('corruptor.call {} called, corruption_level={}'.format(self.name, self._corruptor.corruption_level));        
        return self._corruptor.__call__(inputs);
    
    def _corrupt(self, x):
        self._corruptor.corruption_level = self.corruption_level; # make sure of correct value before call
        log.debug('corruptor.corrupt {} called, corruption_level={}'.format(self.name, self._corruptor.corruption_level));
        return self._corruptor._corrupt(x);
    
