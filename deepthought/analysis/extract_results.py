'''
Created on Apr 30, 2014

@author: sstober
'''
import os;

import logging;
log = logging.getLogger(__name__);

import numpy as np;
from pylearn2.utils import serial
from deepthought.pylearn2ext.util import process_dataset, aggregate_classification;
from deepthought.util.yaml_util import load_yaml_file;

from pylearn2.utils.timing import log_timing;
from sklearn.metrics import confusion_matrix,classification_report;

import fnmatch   


default_check_channels = {
                  'f1': [
                         '_ptrial_mean_f1',
                        '_wtrial_mean_f1',
                        '_pseq_mean_f1',
                        '_wseq_mean_f1',
                        '_f1_mean', 
                       ],
                  'misclass': [
                        '_ptrial_misclass_rate',
                        '_wtrial_misclass_rate',
                        '_trial_misclass_rate',
                        '_pseq_misclass_rate',
                        '_wseq_misclass_rate',
                        '_seq_misclass_rate',
                        '_y_misclass',                                           
                       ],                  
                  } 

test_channel = {
                'f1' : '_f1_mean',
                'misclass' : '_wtrial_misclass_rate'
                };
                  
sort_reverse = {
                'f1' : True,
                'misclass' : False
                };
                         
 
def compare_best_result(experiment_path, mode='f1', check_dataset='valid', print_results=True):
    results = [];
    for root, dirnames, filenames in os.walk(experiment_path):
        for filename in fnmatch.filter(filenames, 'mlp.pkl'):
            value, epochs = extract_best_result(root, mode=mode, check_dataset=check_dataset);
            results.append({
                           'path'  : root,
                           'value' : value,
                           'epochs' : epochs,
                           'test_value' : extract_values(root, [test_channel[mode]], epochs[-1], dataset='test')[0],
                           });
    results = sorted(results, 
                     reverse=sort_reverse[mode],
                     key=lambda entry: entry['value']);
                   
    for i, res in enumerate(results):
        print '{:2}  {} : {:.3f} ({:.3f}) in {}'.format(i, res['path'], res['value'], res['test_value'], res['epochs']);
                     
    return results;

def extract_values(experiment_path, channels, epoch, dataset='valid'):
    model_file = os.path.join(experiment_path, 'mlp.pkl');
    model = serial.load(model_file);
    values = [];
    for channel in channels:
        channel = dataset+channel;
        values.append(model.monitor.channels[channel].val_record[epoch]);
    return values;

def extract_best_result(experiment_path, mode='f1', check_dataset='valid'):
    model_file = os.path.join(experiment_path, 'mlp.pkl');
    model = serial.load(model_file);
    best_results = _extract_best_results(model.monitor.channels, mode=mode, check_dataset=check_dataset);
    best_epochs = _get_best_epochs(best_results); 
    return best_results[0]['value'], best_epochs;
    
def _extract_best_results(channels, mode='f1', check_dataset='valid', check_channels=None):
    print channels.keys();
    
    if check_channels is None:
        check_channels = default_check_channels[mode];
    
    best_results = []; 
    for channel in check_channels:
        channel = check_dataset + channel;
        if not channel in channels:
            continue;
        
        values = channels[channel].val_record;
        
        if mode == 'f1':
            best_value = np.max(values);
        elif mode == 'misclass':
            best_value = np.min(values);
        else:
            raise 'unsupported mode: '+str(mode);
        
        best_results.append( {
                                'name'  : channel,
                                'value' : float(best_value),
                                'epochs' : np.where(values == best_value)[0]
                                } );

    best_results = sorted(best_results, 
                          reverse=sort_reverse[mode],
                          key=lambda entry: entry['value']);
                          
    return best_results;

def _get_best_epochs(best_results):
    best_epochs = best_results[0]['epochs'];
    for entry in best_results:
        print '{:>30} : {:.4f} in {}'.format(entry['name'], entry['value'], entry['epochs']);
        
        intersection = np.intersect1d(best_epochs, entry['epochs']);
        if len(intersection) > 0:
            best_epochs = intersection;
    
    best_epochs = sorted(best_epochs);
    
    return best_epochs;


class DataCube(object):
    
    def __init__(self):
        self.cube = {}; 
        self.store = {};    # alternative associative view
        
    def add(self, metadata, values):
        for meta in metadata:
#             print meta;
            data = values[meta['start']:meta['stop']];
            for cat in ['subject', 'trial_no', 'stimulus', 'channel']:
                self._add_data(cat, meta[cat], data);
            self._add_entries(meta['subject'], meta['stimulus'], meta['channel'], data);
    
    def _add_data(self, cat, key, values):
        if not cat in self.store:
            self.store[cat] = {};
        if not key in self.store[cat]:
            self.store[cat][key] = values;
        else:
            self.store[cat][key] = np.concatenate([self.store[cat][key], values]);
    
    def _add_entries(self, subject, stimulus, channel, values):
        if not subject in self.cube:
            self.cube[subject] = {};
        if not stimulus in self.cube[subject]:
            self.cube[subject][stimulus] = {};
        if not channel in self.cube[subject][stimulus]:
            self.cube[subject][stimulus][channel] = values;
        else:
            self.cube[subject][stimulus][channel] = np.concatenate(
                                    [self.cube[subject][stimulus][channel], values]);
                                    
    def get_store(self):
        return self.store;
    
    def get_entries(self, subjects=None, stimuli=None, channels=None):
        result = [];
        
        if subjects == None:
            subjects = self.cube.keys();
        
        for s in subjects:
            if not s in self.cube:
#                 print 'unknown subject {}'.format(s);
                continue;
            
            if stimuli == None:
                stimuli = self.cube[s].keys();
            
            for r in stimuli:
                if not r in self.cube[s]:
#                     print 'unknown stimulus {}'.format(r);
                    continue;
                
                if channels == None:
                    channels = self.cube[s][r].keys();
                
                for c in channels:
                    if not c in self.cube[s][r]:
#                         print 'unknown channel {}'.format(c);
                        continue;
                
                    result.extend(self.cube[s][r][c]);
        
#         if len(result) > 1:
#             result = np.hstack(result);
        return result;
    
    def get_entries_mean_str(self, subjects=None, stimuli=None, channels=None):
        values = self.get_entries(subjects, stimuli, channels);
        if len(values) > 0:
            return '{:.3f}'.format(np.mean(values));
        else:
            return ' --- ';
        
    def get_table_str(self):
        pass;
    
def extract_output(config, best_epoch):
    # load best model    
    model_file = os.path.join(config.experiment_root, 'epochs', 'epoch{}.pkl'.format(best_epoch));
    print 'loading '+model_file;    
    model = serial.load(model_file);
    
#     print model;

    # additional dataset params
    config.start_sample = 11200;
    config.stop_sample  = 12800;
    config.name = 'test';

    # load dataset    
    dataset, dataset_yaml = load_yaml_file(
                       os.path.join(os.path.dirname(__file__), '..', 'run', 'dataset_template.yaml'),
                       params=config,
                       );    
    
    with log_timing(log, 'processing dataset'):   
        y_real, y_pred, output = process_dataset(model, dataset)
        
    return y_real, y_pred, output;

def _extract_cube(metadata, y_real, y_pred):
    misclass = np.not_equal(y_real, y_pred).astype(int);
    cube = DataCube();
    cube.add(metadata, misclass);
    return cube;

def extract_cube(experiment_path, best_epoch, config):
    
    # load best model    
    model_file = os.path.join(experiment_path, 'epochs', 'epoch{}.pkl'.format(best_epoch));
    print 'loading '+model_file;    
    model = serial.load(model_file);
    
#     print model;

    # additional dataset params
    config.start_sample = 11200;
    config.stop_sample  = 12800;
    config.name = 'test';

    # load dataset    
    dataset, dataset_yaml = load_yaml_file(
                       os.path.join(os.path.dirname(__file__), '..', 'run', 'dataset_template.yaml'),
                       params=config,
                       );    
    
    with log_timing(log, 'processing dataset'):   
        y_real, y_pred, output = process_dataset(model, dataset)
    
    print classification_report(y_real, y_pred);
    print confusion_matrix(y_real, y_pred);
    misclass = np.not_equal(y_real, y_pred).astype(int);
    print misclass.mean();
    
    print '----- sequence aggregration -----'
    s_real, s_pred, s_predf, s_predp = aggregate_classification(
                                          dataset.sequence_partitions,
                                          y_real, y_pred, output);
    
    print classification_report(s_real, s_predf);
    print confusion_matrix(s_real, s_predf);
    print (s_real != s_predf).mean();
    
    print '----- channel aggregration -----'
    t_real, t_pred, t_predf, t_predp = aggregate_classification(
                                          dataset.trial_partitions,
                                          y_real, y_pred, output);
    
    print classification_report(t_real, t_predf);
    print confusion_matrix(t_real, t_predf);
    print (t_real != t_predf).mean();
      
    cube = DataCube();
    cube.add(dataset.metadata, misclass);
    
    for cat, entry in cube.store.items():
        print cat;
        for key, values in cube.store[cat].items():
            print '{:>30} : {:.3f}'.format(key, np.mean(values));
    
    print np.mean(cube.get_entries());
    
    header = '    | ';
    for c in xrange(18):
        header += '  {:2}  '.format(c);
    header += '    avg  ';
    print header;
    
    for r in xrange(48):
        line = '{:>3} | '.format(r); 
        for c in xrange(18):
            line += ' '+cube.get_entries_mean_str(channels=[c], stimuli=[r]);
        line += '   '+cube.get_entries_mean_str(stimuli=[r]); # average over all channels        
        print line;
    
    print
    line = '{:>3} | '.format('avg'); 
    for c in xrange(18):
        line += ' '+cube.get_entries_mean_str(channels=[c]); # average over all stimuli
    line += '   '+cube.get_entries_mean_str(); # average over all stimuli and channels        
    print line;
    
    return cube;  

def extract_results(experiment_path, mode='f1'):
    model_file = os.path.join(experiment_path, 'mlp.pkl');
    model = serial.load(model_file);
    channels = model.monitor.channels;
    
    best_results = _extract_best_results(channels, mode);    
    best_epochs = _get_best_epochs(best_results);
    
    print 'overall best epochs: {}'.format(best_epochs);
    
    #     best_epoch = best_results[0]['epoch'];
    best_epoch = best_epochs[-1]; # take last entry -> more stable???
    
    # double check
    assert best_epoch == channels['valid_y_misclass'].epoch_record[best_epoch]
    
    print 'test values for best epoch:'
    results = {};
    results['best_epoch'] = best_epoch;
    for channel in [
                    'test_ptrial_misclass_rate',
                    'test_wtrial_misclass_rate',
                    'test_trial_misclass_rate',
                    
                    'test_pseq_misclass_rate',
                    'test_wseq_misclass_rate',
                    'test_seq_misclass_rate',
                    
                    'test_y_misclass',
                    
                    'test_ptrial_mean_f1',
                    'test_wtrial_mean_f1',
                    'test_trial_mean_f1',
                    
                    'test_pseq_mean_f1',
                    'test_wseq_mean_f1',
                    'test_seq_mean_f1',
                    
                    'test_f1_mean',
#                     'test_confusion_African_as_Western',
#                     'test_confusion_Western_as_African',
                    ]:
        
        if channel in channels:
            value = float(channels[channel].val_record[best_epoch]);
        else:
            value = np.NAN;
        print '{:>30} : {:.4f}'.format(channel, value);
        results[channel] = value;

    # add meta-channels:
    results['frame_misclass'] = results['test_y_misclass']
    
    strategy = 'threshold';
#     strategy = 'min_valid';
    
    if strategy == 'threshold':
        # decide based on validation set, which channel to use
        if channels['valid_y_misclass'].val_record[best_epoch] > 0.52: 
            # less than 60% accuracy
            results['sequence_misclass'] = channels['test_seq_misclass_rate'].val_record[best_epoch];
            results['trial_misclass'] = channels['test_trial_misclass_rate'].val_record[best_epoch];
        else:
            # at least 60% accuracy => use weighted version
            results['sequence_misclass'] = channels['test_wseq_misclass_rate'].val_record[best_epoch];
            results['trial_misclass'] = channels['test_wtrial_misclass_rate'].val_record[best_epoch];

    else:    
        # decide based on validation set, which channel to use
        def get_best_value(check_channels):
            values = np.zeros(len(check_channels));
            for i, channel in enumerate(check_channels):
                values[i] = channels['valid'+channel].val_record[best_epoch];
                
                test = channels['test'+channel].val_record[best_epoch];
                print '{:>22} valid {:.3f}  test {:.3f}'.format(channel, values[i], test);
                
            best_i = np.argmin(values);
            best = channels['test'+check_channels[best_i]].val_record[best_epoch];
            print 'best @ {} -> test value: {:.3f}'.format(best_i, best);
            return best
        
        results['sequence_misclass'] = get_best_value([
                                        '_seq_misclass_rate',
                                        '_wseq_misclass_rate',
                                        '_pseq_misclass_rate',                            
                                        ]);
                                        
        results['trial_misclass'] = get_best_value([
                                        '_trial_misclass_rate',
                                        '_wtrial_misclass_rate',
                                        '_ptrial_misclass_rate',                            
                                        ]);
    
    
    # this would be cheating
#     results['sequence_misclass'] = np.min([results['test_pseq_misclass_rate'],
#                                            results['test_wseq_misclass_rate'],
#                                            results['test_seq_misclass_rate']]);
#     results['trial_misclass'] = np.min([results['test_ptrial_misclass_rate'],
#                                            results['test_wtrial_misclass_rate'],
#                                            results['test_trial_misclass_rate']]);    

    return results