'''
Created on Apr 17, 2014

@author: sstober

@deprecated: replaced by scripts in run/ that use more YAML
'''

import time;
import logging;

import os;
import numpy as np;

log = logging.getLogger(__name__);

from deepthought.experiments.ismir2014.util import load_config;
from deepthought.util.yaml_util import load_yaml_file, save_yaml_file;
from deepthought.util.config_util import merge_params;

from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing

from pylearn2.datasets.transformer_dataset import TransformerDataset;
from pylearn2.blocks import StackedBlocks;
from pylearn2.training_algorithms.sgd import SGD;
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError;
from pylearn2.termination_criteria import EpochCounter;
from pylearn2.train import Train;
from pylearn2.corruption import BinomialCorruptor;
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor;
from pylearn2.training_algorithms.sgd import OneOverEpoch;

from deepthought.pylearn2ext.util import LoggingCorruptor, LoggingCallback;

from deepthought.pylearn2ext import StackedDenoisingAutoencoder

from deepthought.experiments.ismir2014.plot import scan_for_best_performance;

from deepthought.experiments.ismir2014.extract_results import extract_results;

import numpy

def get_layer_trainer_sgd_autoencoder(
                                      layer, 
                                      trainset,
                                      batch_size = 10,                                      
                                      learning_rate=0.1, 
                                      max_epochs=100, 
                                      name=''):    
    # configs on sgd
    train_algo = SGD(
            learning_rate = learning_rate,
#             learning_rule = AdaDelta(),
            learning_rule = Momentum(init_momentum=0.5),
              cost =  MeanSquaredReconstructionError(),
              batch_size = batch_size,
              monitoring_dataset =  trainset,
              termination_criterion = EpochCounter(max_epochs=max_epochs),
              update_callbacks = None
              )
    
    log_callback = LoggingCallback(name);
    
    return Train(model = layer,
            algorithm = train_algo,
            extensions = [log_callback, 
                          OneOverEpoch(start=1, half_life=5),
                          MomentumAdjustor(final_momentum=0.7, start=10, saturate=100)],
            dataset = trainset)

def train_sda(params):
    
    input_trainset, trainset_yaml_str = load_yaml_file(
                   os.path.join(os.path.dirname(__file__), 'train_sda_dataset_template.yaml'),
                   params=params,
                   );
    
    log.info('... building the model');
                       
    # build layers
    layer_dims = [params.input_length];
    layer_dims.extend(params.hidden_layers_sizes);
        
    layers = [];
    for i in xrange(1, len(layer_dims)):                   
        layer_params = {
                'name' : 'da'+str(i),
                'n_inputs' : layer_dims[i-1],
                'n_outputs' : layer_dims[i],
                'corruption_level' : params.pretrain.corruption_levels[i-1],
                'input_range' : numpy.sqrt(6. / (layer_dims[i-1] + layer_dims[i])),
                'random_seed' : params.random_seed, 
                }
        
        layers.append(load_yaml_file(
                           os.path.join(os.path.dirname(__file__), 'train_sda_layer_template.yaml'),
                           params=layer_params,
                           )[0]);
    
    # unsupervised pre-training
    log.info('... pre-training the model');
    start_time = time.clock();  
    
    for i in xrange(len(layers)):
        # reset corruption to make sure input is not corrupted
        for layer in layers:
            layer.set_corruption_level(0);
            
        if i == 0:
            trainset = input_trainset;
        elif i == 1:
            trainset = TransformerDataset( raw = input_trainset, transformer = layers[0] );
        else:
            trainset = TransformerDataset( raw = input_trainset, transformer = StackedBlocks( layers[0:i] ));
            
        # set corruption for layer to train
        layers[i].set_corruption_level(params.pretrain.corruption_levels[i]);
        
        # FIXME: this is not so nice but we have to do it this way as YAML is not flexible enough
        trainer = get_layer_trainer_sgd_autoencoder(
                        layers[i], 
                        trainset,
                        learning_rate       = params.pretrain.learning_rate,
                        max_epochs          = params.pretrain.epochs,
                        batch_size          = params.pretrain.batch_size,                       
                        name='pre-train'+str(i));
        
        log.info('unsupervised training layer %d, %s '%(i, layers[i].__class__));
        trainer.main_loop();
        
#         theano.printing.pydotprint_variables(
#                                      layer_trainer.algorithm.sgd_update.maker.fgraph.outputs[0],
#                                      outfile='pylearn2-sgd_update.png',
#                                      var_with_name_simple=True);
        
    end_time = time.clock();
    log.info('pre-training code ran for {0:.2f}m'.format((end_time - start_time) / 60.));
    
    if params.untie_weights:
        # now untie the decoder weights
        log.info('untying decoder weights');
        for layer in layers:
            layer.untie_weights();
    
    # construct multi-layer training functions
    
    # unsupervised training
    log.info('... training the model');
    
    sdae = None;
    for depth in xrange(1, len(layers)+1):
        first_layer_i = len(layers)-depth;
        log.debug('training layers {}..{}'.format(first_layer_i,len(layers)-1));

        group = layers[first_layer_i:len(layers)];
#         log.debug(group);
        
        # reset corruption 
        for layer in layers:
            layer.set_corruption_level(0);
                
        if first_layer_i == 0:
            trainset = input_trainset;
        elif first_layer_i == 1:
            trainset = TransformerDataset( raw = input_trainset, transformer = layers[0] );
        else:
            trainset = TransformerDataset( raw = input_trainset, transformer = StackedBlocks( layers[0:first_layer_i] ));
            
        # set corruption for input layer of stack to train
#         layers[first_layer_i].set_corruption_level(stage2_corruption_levels[first_layer_i]);

        corruptor = LoggingCorruptor(
                        BinomialCorruptor(
                            corruption_level=params.pretrain_finetune.corruption_levels[first_layer_i]),
                        name='depth {}'.format(depth));
        sdae = StackedDenoisingAutoencoder(group, corruptor);      
             
        trainer = get_layer_trainer_sgd_autoencoder(
                                    sdae,
                                    trainset, 
                                    learning_rate       = params.pretrain_finetune.learning_rate,
                                    max_epochs          = params.pretrain_finetune.epochs,
                                    batch_size          = params.pretrain_finetune.batch_size,
                                    name='multi-train'+str(depth)
                                    );
                                    
        log.info('unsupervised multi-layer training %d'%(i));        
        trainer.main_loop()
    
    end_time = time.clock()
    log.info('full training code ran for {0:.2f}m'.format((end_time - start_time) / 60.));        
    
    # save the model
    model_file = os.path.join(params.experiment_root, 'sda', 'sda_all.pkl'); 
    with log_timing(log, 'saving SDA model to {}'.format(model_file)):
        serial.save(model_file, sdae);

    if params.untie_weights:
        # save individual layers for later (with untied weights)
        for i, layer in enumerate(sdae.autoencoders):
            layer_file = os.path.join(params.experiment_root, 'sda', 'sda_layer{}_untied.pkl'.format(i));
            with log_timing(log, 'saving SDA layer {} model to {}'.format(i, layer_file)):
                serial.save(layer_file, layer);
            
    # save individual layers for later (with tied weights)
    for i, layer in enumerate(sdae.autoencoders):
        if params.untie_weights:
            layer.tie_weights(); 
        layer_file = os.path.join(params.experiment_root, 'sda', 'sda_layer{}_tied.pkl'.format(i));
        with log_timing(log, 'saving SDA layer {} model to {}'.format(i, layer_file)):
            serial.save(layer_file, layer);        

    log.info('done');
    
    return sdae;

def train_mlp(params):
    
#     sda_file = os.path.join(params.experiment_root, 'sda', 'sda_all.pkl');

    # check whether pre-trained SDA is there
    pretrained = True;
    for i in xrange(len(params.hidden_layers_sizes)):
        sda_layer_file = params.get(('layer{}_content').format(i));
        if not os.path.isfile(sda_layer_file):
            log.info('did not find pre-trained SDA layer model at {}. re-computing SDA'.format(sda_layer_file));
            pretrained = False;
            break;
        else:
            log.info('found pre-trained SDA layer model at {}'.format(sda_layer_file));
    
    if not pretrained:
        train_sda(params);
        
    n_layers = len(params.hidden_layers_sizes);
        
    if params.learning_rule == 'AdaDelta':
        yaml_template = 'train_sda_mlp_template.AdaDelta.yaml'
    else:
        if n_layers == 3:
            yaml_template = 'train_sda_mlp_template.Momentum.yaml'
        elif n_layers == 2:
            yaml_template = 'train_sda_mlp_template.Momentum.2layers.yaml'
        else:
            raise '{} layers not supported'.format(n_layers);
    
    train, train_yaml_str = load_yaml_file(
                   os.path.join(os.path.dirname(__file__), yaml_template),
                   params=params,
                   );
                   
    save_yaml_file(train_yaml_str, os.path.join(params.experiment_root, 'mlp_train.yaml'));
    
    with log_timing(log, 'training MLP'):    
        train.main_loop();
        
    log.info('done');
    
def get_default_config_path():
    return os.path.join(os.path.dirname(__file__),'train_sda_mlp.cfg');

if __name__ == '__main__':
#     config = load_config(default_config='../../train_sda.cfg', reset_logging=False);
    config = load_config(default_config=get_default_config_path(), reset_logging=False);
                         
    hyper_params = {   
    };
    
    params = merge_params(config, hyper_params);

    if not config.get('only_extract_results', False):
        train_mlp(params);
        
    scan_for_best_performance(params.experiment_root, 'valid_y_misclass');
    scan_for_best_performance(params.experiment_root, 'valid_ptrial_misclass_rate')
    
    values = extract_results(config.experiment_root, mode='misclass');        
            
    print np.multiply(100, [
#                         1 - values['test_y_misclass'],
#                         1 - values['test_wseq_misclass_rate'],
#                         1 - values['test_wtrial_misclass_rate']]);     
               
                1 - values['frame_misclass'],
                1 - values['sequence_misclass'],
                1 - values['trial_misclass']]);