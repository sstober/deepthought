'''
Created on Apr 12, 2014

@author: sstober
'''

import os;

import logging;
log = logging.getLogger(__name__);

from pylearn2.utils.timing import log_timing

import numpy as np;

from deepthought.experiments.ismir2014.util import load_config, save;

from pylearn2.utils import serial;
from pylearn2.config import yaml_parse

import theano;
import theano.tensor as T;
from sklearn.metrics import confusion_matrix,classification_report;

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter;
from pylearn2.space import CompositeSpace


def analyze(config):
    output_path = config.get('output_path');
#     model_file = os.path.join(output_path, 'eeg', 'conv3', 'convolutional_network.pkl');
#     model_file = os.path.join(output_path, 'eeg', 'conv10', 'epochs', 'cnn_epoch94.pkl');
    model_file = '../../../debug/debug_run4/debug_network.pkl';
    with log_timing(log, 'loading convnet model from {}'.format(model_file)):
        model = serial.load(model_file);
        
    input_shape =  model.get_input_space().shape;
        
    config = config.eeg;
    hyper_params = {
                'input_length':input_shape[0], #25+151-1+301-1, # this should leave a single value per channel after convolution
                'hop_size':5,               # reduce amount of data by factor 5
                
                'dataset_root': config.get('dataset_root'),
                'dataset_suffix': config.get('dataset_suffix'),
                'save_path': config.get('save_path'),
        }
        
    dataset_yaml = '''
    !obj:deepthought.datasets.rwanda2013rhythms.EEGDataset.EEGDataset {
                                 name : 'testset',
                                 path : %(dataset_root)s, 
                                 suffix : '_channels', # %(dataset_suffix)s,
                                 subjects : [0],
                                 resample : [400, 100],
                                 start_sample : 2500,
                                 stop_sample  : 3200,     # None (empty) = end of sequence
                  # FIXME:                
#                                  n_fft : 24,
#                                  frame_size : 10, # %(input_length)i,                                
                                 frame_size : %(input_length)i,
                                 
                                 hop_size : %(hop_size)i,           
                                 label_mode : 'rhythm_type',
#                                  save_matrix_path: '../../../debug/debug.pkl'
                            }
'''
    dataset_yaml = dataset_yaml  % hyper_params;
    print dataset_yaml;

    with log_timing(log, 'parsing yaml'):    
        testset = yaml_parse.load(dataset_yaml);
        
#     print testset.subject_partitions;
#     print testset.sequence_partitions;
    
    seq_starts = testset.sequence_partitions;
#     return;
    
#     axes=['b', 0, 1, 'c']
#     def dimshuffle(b01c):
#         default = ('b', 0, 1, 'c')
#         return b01c.transpose(*[default.index(axis) for axis in axes])
#     data = dimshuffle(testset.X);
    
#     design_matrix = model.get_design_matrix()

#     view_converter = DefaultViewConverter([475, 1, 1]);
#     data = view_converter.


#     ## get the labels
#     data_specs= (model.get_output_space(), "targets");
#     it = testset.iterator(
#                            mode='sequential', 
#                            batch_size=100,
#                            data_specs=data_specs);
#     labels = np.hstack([np.argmax(minibatch, axis = 1) for minibatch in it])
#     print labels[0:1000]
# 
#     ## get the predictions
#     minibatch = model.get_input_space().make_theano_batch();
#     output_fn = theano.function(inputs=[minibatch], 
#                                 outputs=T.argmax(model.fprop(minibatch), axis = 1));
#     print "function compiled"
# #     data_specs= (CompositeSpace((
# #                                 model.get_input_space(), 
# #                                 model.get_output_space())), 
# #                 ("features", "targets"));
#                 
#     data_specs= (model.get_input_space(), "features");    
#     it = testset.iterator(
#                             mode='sequential', 
#                             batch_size=100,
#                             data_specs=data_specs);
#     print "iterator ready"
#         
#     y_pred = np.hstack([output_fn(minibatch) for minibatch in it])
#     
#     print y_pred[0:1000]
    
    
    minibatch = model.get_input_space().make_theano_batch();
    output_fn = theano.function(inputs=[minibatch], 
                                outputs=T.argmax(model.fprop(minibatch), axis = 1));
    print "function compiled"
    
    data_specs= (CompositeSpace((
                                model.get_input_space(), 
                                model.get_output_space())), 
                ("features", "targets"));
    it = testset.iterator('sequential',
                          batch_size=100,
                          data_specs=data_specs);
    print "iterator ready"
                    
    y_pred = [];
    y_real = [];                
    for minibatch, target in it:
        y_pred.append(output_fn(minibatch));
        y_real.append(np.argmax(target, axis = 1));
    y_pred = np.hstack(y_pred);
    y_real = np.hstack(y_real);   
    
    print y_pred[0:1000]
    
    print classification_report(y_real, y_pred);
    print confusion_matrix(y_real, y_pred);

    misclass = (y_real != y_pred);
    print misclass.mean();
    
    correct = 0;
    s_real = [];
    s_pred = [];
    s_pred_agg = [];
    
    n_channels = 16;
    channel_scores = np.zeros(n_channels, dtype=np.int);
    
    for i in xrange(len(seq_starts)):
        
        start = seq_starts[i];
        if i < len(seq_starts) - 1:
            stop = seq_starts[i+1];
        else:
            stop = None;
        
        s_real.append(y_real[start]);
        
#         print np.bincount(y_pred[start:stop]);
#         print np.argmax(np.bincount(y_pred[start:stop]));

        s_pred.append(np.argmax(np.bincount(y_pred[start:stop])));
        
        s_pred_agg.append(np.mean(y_pred[start:stop])); # works only for binary classification
        
        seq_misclass = misclass[start:stop].mean();
#         print '{} [{}{}]: {}'.format(i, start, stop, seq_misclass);
        
        if seq_misclass < 0.5: # more correct than incorrect
            correct += 1;
            channel_scores[i%n_channels] += 1;
    
    s_real = np.hstack(s_real);
    s_pred = np.hstack(s_pred);  
    
    print s_real;
    print s_pred;       
    print s_pred_agg;
    
    print 'aggregated'
    print classification_report(s_real, s_pred);
    print confusion_matrix(s_real, s_pred);
    
    s_misclass = (s_real != s_pred);
    print s_misclass.mean();
    
    print channel_scores;
    
    return;
    
    
    
    
    
    
    
    

    input_shape =  model.get_input_space().shape;
    
    print input_shape
    
    view_converter = DefaultViewConverter((input_shape[0], input_shape[1], 1));
    
    data = view_converter.design_mat_to_topo_view(testset.X);
    print data.shape;
                
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop( X )
    Y = T.argmax( Y, axis = 1 ) # needed - otherwise not single value
    output_fn = theano.function( [X], Y );
    


    
#     y_pred = output_fn( data );

    batch_size = 1000;
    y_pred = [];
    batch_start = 0;
    while batch_start < data.shape[0]:
        batch_stop = min(data.shape[0], batch_start + batch_size);
        y_pred.append(output_fn( data[batch_start:batch_stop] ));
#         if batch_start == 0: print y_pred;
        batch_start = batch_stop;
    y_pred = np.hstack(y_pred);

    print testset.labels[0:1000]
    print y_pred[0:1000]

    print classification_report(testset.labels, y_pred);
    print confusion_matrix(testset.labels, y_pred);

    labels = np.argmax(testset.y, axis=1)
    print classification_report(labels, y_pred);
    print confusion_matrix(labels, y_pred);
    
    labels = np.argmax(testset.y, axis=1)
    print classification_report(labels, y_pred);
    print confusion_matrix(labels, y_pred);

    misclass = (labels != y_pred).mean()
    print misclass
    
#     # alternative version from KeepBestParams
#     minibatch = T.matrix('minibatch')
#     output_fn = theano.function(inputs=[minibatch],outputs=T.argmax( model.fprop(minibatch), axis = 1 ));
#     it = testset.iterator('sequential', batch_size=batch_size, targets=False);
#     y_pred = [output_fn(mbatch) for mbatch in it];

#             y_hat = T.argmax(state, axis=1)
#             y = T.argmax(target, axis=1)
#             misclass = T.neq(y, y_hat).mean()
#             misclass = T.cast(misclass, config.floatX)
#             rval['misclass'] = misclass
#             rval['nll'] = self.cost(Y_hat=state, Y=target)
        
    

    log.debug('done');
    


if __name__ == '__main__':
    config = load_config(default_config='../train_sda.cfg', reset_logging=False);    
    analyze(config);