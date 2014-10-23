#!/usr/bin/env python
'''
Created on May 12, 2014

@author: sstober
'''
import os
import argparse
import socket

import logging;
from deepthought.datasets.rwanda2013rhythms.LabelConverter import LabelConverter, swapped_meta_labels
from deepthought.datasets.rwanda2013rhythms.PathLocalizer import PathLocalizer;
log = logging.getLogger(__name__);

import numpy as np
import matplotlib.pyplot as plt;
import matplotlib as mpl
from sklearn.metrics import confusion_matrix,classification_report

from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing

from deepthought.util.config_util import init_logging;
from deepthought.util.yaml_util import load_yaml_template, load_yaml;
from deepthought.util.fs_util import ensure_parent_dir_exists;
from deepthought.pylearn2ext.util import process_dataset;

def multi_level_accuracy_analysis(y_real, y_pred):
    shuffle_classes = LabelConverter().shuffle_classes;
    
    def transform(x, mapping):
        return [int(mapping[v]) for v in x];
    
    #print (y_real != y_pred).mean();
    
    #print shuffle_classes
    y_real_shuffled = transform(y_real, shuffle_classes);
    y_pred_shuffled = transform(y_pred, shuffle_classes);
    #print y_real[0:10];
    #print y_real_shuffled[0:10];
    acc24 = 100 * (1 - np.not_equal(y_real_shuffled, y_pred_shuffled).mean());
    print '{:.2f}% rhythm level accuracy (24 classes, chance level {:.2f}%)'.format(acc24, 100/24.);
    
    rhythm_pairs_mapping = [i/2 for i in xrange(24)];
    #print rhythm_pairs_mapping
    y_real_pairs = transform(y_real_shuffled, rhythm_pairs_mapping);
    y_pred_pairs = transform(y_pred_shuffled, rhythm_pairs_mapping);
    #print y_real_pairs
    #print y_pred_pairs
    acc12 = 100 * (1 - np.not_equal(y_real_pairs, y_pred_pairs).mean());
    print '{:.2f}% rhythm pair level accuracy (12 classes, chance level {:.2f}%)'.format(acc12, 100/12.);
    
    rhythm_group_mapping = [i/6 for i in xrange(24)];
    #print rhythm_group_mapping
    y_real_groups = transform(y_real_shuffled, rhythm_group_mapping);
    y_pred_groups = transform(y_pred_shuffled, rhythm_group_mapping);
    #print y_real_groups
    #print y_pred_groups
    acc4 = 100 * (1 - np.not_equal(y_real_groups, y_pred_groups).mean());
    print '{:.2f}% rhythm group level accuracy (4 classes, chance level {:.2f}%)'.format(acc4, 25);
    
    rhythm_type_mapping = [i/12 for i in xrange(24)];
    #print rhythm_type_mapping
    y_real_types = transform(y_real_shuffled, rhythm_type_mapping);
    y_pred_types = transform(y_pred_shuffled, rhythm_type_mapping);
    #print y_real_types
    #print y_pred_types
    acc2 = 100 * (1 - np.not_equal(y_real_types, y_pred_types).mean());
    print '{:.2f}% rhythm type level accuracy (2 classes, chance level {:.2f}%)'.format(acc2, 50);
    
    return acc24, acc12, acc4, acc2;

def generate_plots(y_real, y_pred, output, dataset_name, output_path):
    font = {
#         'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

    mpl.rc('font', **font)

    # Compute confusion matrix
    cm = confusion_matrix(y_real, y_pred);
    
    shuffle_classes = LabelConverter().shuffle_classes;
    cmt = np.zeros([24,24]);
    for i in xrange(24):
        for j in xrange(24):
            cmt[shuffle_classes[i],shuffle_classes[j]] = cm[i,j];
    labels = swapped_meta_labels;

    print cmt  
    print classification_report(y_real, y_pred)
    
    misclass = (y_real != y_pred).mean();
    print 'misclassification rate: {:.4f}'.format(misclass);
    print 'accuracy: {:.4f}'.format(100. * (1 - misclass));

    fig = plt.figure(1, figsize=(10, 10), dpi=600)
    axes = fig.add_subplot(111)
    
    # Show confusion matrix in a separate 
    axes.matshow(cmt)
#     plt.title('Confusion matrix')
#     axes.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    axes.set_xticks(xrange(24));
    axes.set_xticklabels(labels, rotation=90, fontsize=15)
    axes.set_yticks(xrange(24));
    axes.set_yticklabels(labels, fontsize=15)
    
    plot_file = os.path.join(experiment_root, 'confusion', '{}_confusion.pdf'.format(dataset_name));
    ensure_parent_dir_exists(plot_file);
    plt.savefig(plot_file, bbox_inches='tight')
#     plt.show()

def load_results(experiment_root):
    # load the model (mlp_best.pkl)
    model_file = os.path.join(experiment_root, 'mlp_best.pkl');    
    with log_timing(log, 'loading model from {}'.format(model_file)):  
        model = serial.load(model_file);    

    # load train
    train_yaml_file = os.path.join(experiment_root, 'train.yaml');
    train_yaml = load_yaml_template(train_yaml_file);
    
    # fix dataset path
    localizer = PathLocalizer();
    train_yaml = localizer.localize_yaml(train_yaml);
    
    with log_timing(log, 'loading train from {}'.format(train_yaml_file)):      
        train = load_yaml(train_yaml)[0];
    
    return train, model;

if __name__ == '__main__':
    init_logging(pylearn2_loglevel=logging.INFO);
    parser = argparse.ArgumentParser(prog='generate_plots', 
                                     description='generates plots ;-)');
     
    # global options
    parser.add_argument('path', help='root path of the experiment');
     
    args = parser.parse_args();
          
    experiment_root = args.path;

#     experiment_root = '/Users/sstober/git/deepbeat/deepbeat/spearmint/h0_input47/20041_h0_pattern_width_[47]_h0_patterns_[30]_h0_pool_size_[1]_learning_rate_[0.01]'
#     path = '/Users/sstober/git/deepbeat/deepbeat/spearmint/best/h0_1bar_nophase_49bins'

    train, model = load_results(experiment_root);
        
    # get the datasets with their names from the monitor
    for key, dataset in train.algorithm.monitoring_dataset.items():
        # process each dataset 
        with log_timing(log, 'processing dataset \'{}\''.format(key)): 
            y_real, y_pred, output = process_dataset(model, dataset)
            
            generate_plots(y_real, y_pred, output, key, experiment_root);
    
    
