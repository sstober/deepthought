'''
Created on Apr 4, 2014

@author: sstober
'''
import os;

import logging;
from deepthought.util.config_util import merge_params
log = logging.getLogger(__name__);

import numpy as np;
import theano;
import matplotlib.pyplot as plt;

from pylearn2.utils.timing import log_timing

from deepthought.experiments.ismir2014.util import load_config, save;

from pylearn2.utils import serial;

from deepthought.util.yaml_util import load_yaml_file;


def reconstruct_time_series(frames, hop_size):
    n_frames    = frames.shape[0];
    frame_size  = frames.shape[1];
    
    y           = np.zeros(frame_size + hop_size * (n_frames - 1))
    
#     window = 1.0 / 12.0; # 12 overlapping windows
    window = hop_size / float(frame_size);
    
    log.debug('#frames: {}\t frame_length: {}\t scaling: {}'.format(n_frames, frame_size, window));

    for i in xrange(n_frames):
        sample  = i * hop_size
        frame   = frames[i];
        ytmp    = window * frame;

        y[sample:(sample+frame_size)] = y[sample:(sample+frame_size)] + ytmp

    return y

# overview of frame plots
def multiplot(data, caption='', yrange=None, file_path=None, with_titles=False):
    print 'plotting {}'.format(caption);
    plt.figure();    
    plt.title(caption);
    for i in xrange(len(data)):
        plt.subplot(10,10,i+1);
        if with_titles:
            plt.title(i);
        frame1 = plt.gca();    
        frame1.axes.get_xaxis().set_visible(False);
        frame1.axes.get_yaxis().set_visible(False);
        if yrange is not None:
            plt.ylim(yrange);
        plt.plot(data[i]);
    plt.ioff();
    
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, dpi=600)   

def analyze_frames(dataset, output_fn):
    inputs = dataset.X[0:100]; # .get_value(borrow=True);
    outputs = output_fn(inputs);
    sample_error = np.abs(inputs-outputs);

    multiplot(inputs, 'Input', yrange=[-1,1], );        
    multiplot(outputs, 'Reconstruction', yrange=[-1,1]);    
    multiplot(sample_error, 'Error', yrange=[0,2]);
    
def analyze_worst_frames(dataset, output_fn, num_worst=100, output_path=None):
    inputs = dataset.X; # .get_value(borrow=True);
    outputs = output_fn(inputs);
    loss = ((inputs - outputs) ** 2).sum(axis=1); # from MeanSquaredReconstructionError - without mean()
    
    worst_i = np.argsort(loss)[::-1][:num_worst]; # [::-1] reverses the array returned by argsort() and [:n] gives that last n elements
#     print worst_i;
    
    worst_error = [loss[i] for i in worst_i];
    worst_inputs = np.vstack([inputs[i] for i in worst_i]);
    worst_outputs = np.vstack([outputs[i] for i in worst_i]);
    worst_sample_error = np.abs(worst_inputs-worst_outputs);

#     print worst_error;
#     print worst_inputs.shape;
#     print worst_outputs.shape; 
    
    if output_path is None:
        output_path = '/Users/sstober/git/deepbeat/deepbeat/plot/';
    
    multiplot(worst_inputs, 'Input', yrange=[-1,1], file_path=os.path.join(output_path,output_path,'worst_input.pdf'));        
    multiplot(worst_outputs, 'Reconstruction', yrange=[-1,1], file_path=os.path.join(output_path,output_path,'worst_output.pdf'));      
    multiplot(worst_sample_error, 'Error', yrange=[0,2], file_path=os.path.join(output_path,output_path,'worst_delta.pdf'));
    save(os.path.join(output_path, 'worst_plotdata.pklz'), [worst_inputs, worst_outputs, worst_sample_error, worst_error, worst_i]);
      
    
def analyze_reconstruction(dataset, output_fn, hop_size):
    #     inputs = dataset.X[0:2122]; # .get_value(borrow=True);
    inputs = dataset.X[0:522]; # .get_value(borrow=True);
    outputs = output_fn(inputs);

    x = reconstruct_time_series(inputs, hop_size=hop_size);
    y = reconstruct_time_series(outputs, hop_size=hop_size);
    e = np.abs(x-y);
    
    print len(y);
    
    plt.figure();    
#     plt.title(caption);
#     for i in xrange(len(y)):
#         plt.subplot(10,10,i+1);
#         plt.title(i);
#         frame1 = plt.gca();    
#         frame1.axes.get_xaxis().set_visible(False);
#         frame1.axes.get_yaxis().set_visible(False);
#         if yrange is None:
    plt.ylim([-1,1]);        
    plt.plot(x);
#     plt.ioff();
    plt.show();

def analyze_complex(dataset, output_fn, hop_size, output_path=None):
#     inputs = dataset.X[0:2122]; # .get_value(borrow=True);

    seq_id = 13;
    
    start_of_sequence = dataset.sequence_partitions[seq_id];
    end_of_sequence = dataset.sequence_partitions[seq_id+1];
    inputs = dataset.X[start_of_sequence:end_of_sequence];
    outputs = output_fn(inputs);
    sample_error = np.abs(inputs-outputs);
    
    sre = ((inputs - outputs) ** 2).sum(axis=1);
    print sre.mean();

    x = reconstruct_time_series(inputs, hop_size=hop_size);
    y = reconstruct_time_series(outputs, hop_size=hop_size);
    e = np.abs(x-y);
    
    print len(y);

    #fig, ax = ppl.subplots(1)

    #ppl.pcolormesh(fig, ax, sample_error[0:100]);
    #fig.show();

    if output_path is None:
        output_path = '/Users/sstober/git/deepbeat/deepbeat/plot/';

    # to be used in ipython notebook
    save(os.path.join(output_path, 'plotdata.pklz'), [inputs, outputs, x, y, sre]);
    
def analyze_reconstruction_error(datasets, output_fn):
    report = [];
    for dataset_name, dataset in datasets.items():
#         start_of_sequence = 0;
#         for end_of_sequence in dataset.sequence_partitions:
#             inputs = dataset.X[start_of_sequence:end_of_sequence];
#             outputs = output_fn(inputs);
#             start_of_sequence = end_of_sequence;
        inputs = dataset.X; 
        outputs = output_fn(inputs);
        loss = ((inputs - outputs) ** 2).sum(axis=1); # from MeanSquaredReconstructionError - without mean()
        
        report.append([dataset_name, loss.mean(), loss.std(), loss.max()]);
    
    return report;

# define datasets to use
dataset_params = {
                  'valid' : {
                             'name' : 'valid',
                             'start_sample' : 0,
                             'stop_sample' : 1600,                                  
                             },
                  'train' : {
                             'name' : 'train',
                             'start_sample' : 1600,
                             'stop_sample' : 11200, 
                             },
                  'test' : {
                             'name' : 'test',
                             'start_sample' : 11200,
                             'stop_sample' : 12800, 
                             },
                  'post' : {
                             'name' : 'post',
                             'start_sample' : 12800,
                             'stop_sample' : 13600, 
                             }
                  };

def analyze_msre(config):
    report = analyze(config, dataset_names=['train','test'], subjects='all', worst_frames=False);
    result = {};
    for line in report:
        print 'MSRE for {}: {}'.format(line[0], line[1]);
        result[line[0]] = line[1];
    return result;

def analyze(config, dataset_names=['train','test'], subjects=None, worst_frames=False):
    model_file = os.path.join(config.experiment_root, 'sda', 'sda_all.pkl');
#     output_path = config.tempo.get('output_path');
#     model_file = os.path.join(output_path, 'sdae-model.pkl');
#     model_file = '/Users/sstober/git/deepbeat/deepbeat/output/gpu.old/sda/exp6.14all/sda/sda_all.pkl';
#     model_file = '/Users/sstober/git/deepbeat/deepbeat/output/sda-mlp6/sda/sda_all.pkl';

    output_path = os.path.join(os.path.dirname(model_file), 'plot');
    if not os.path.exists(output_path):
        os.mkdir(output_path);

    with log_timing(log, 'loading SDA model from {}'.format(model_file)):
        sda = serial.load(model_file);    
    
    # prepare to modify subjects parameter    
    if subjects is None:              
        subjects = config.subjects;
    config.subjects = None;
        
    ## load datasets for subject
    def load_datasets_for_subjects(dataset_params, subjects, suffix=''):
        datasets = {}
        for key, params in dataset_params.items():
            if not key in dataset_names:
                continue;            
            params['subjects'] = subjects;
            params['name'] = params['name']+suffix;
            dataset_config = merge_params(config, params);
            dataset, dataset_yaml = load_yaml_file(
                       os.path.join(os.path.dirname(__file__), 'run', 'dataset_template.yaml'),
                       params=dataset_config,
                       );
    #        log.info('dataset loaded. X={} y={}'.format(dataset.X.shape, dataset.y.shape));
            datasets[key+suffix] = dataset;
            del dataset, dataset_yaml;
        return datasets;
    
    datasets = load_datasets_for_subjects(dataset_params, subjects);
#     full_datasets = load_datasets_for_subjects(dataset_params, 'all'); # , suffix='all');
        
    # compile function
    X = sda.get_input_space().make_theano_batch()
    Y = sda.output( X )    
    output_fn = theano.function( [X], Y );
    
    def minibatched_output_fn(input):
        input_size = len(input);
        start = 0;
        output = [];
        while start < input_size:
            stop = np.min([start+1000, input_size]);
            output.append(output_fn(input[start:stop]));
            start = stop;
        return np.vstack(output);
    
    report = [];
    report = analyze_reconstruction_error(datasets, minibatched_output_fn);
    
    for line in report:
        print '{:>6} : mean = {:.3f} std = {:.3f} max = {:.3f}'.format(*line);
    
#     analyze_frames(dataset, output_fn);    

    if worst_frames:
        analyze_worst_frames(datasets['train'], minibatched_output_fn, output_path=output_path);
     
#     analyze_reconstruction(dataset, minibatched_output_fn, hop_size);
    
    analyze_complex(datasets['test'], minibatched_output_fn, config.hop_size, output_path=output_path);

    return report;
    

if __name__ == '__main__':
    config = load_config(default_config=
#                          '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_400Hz_80-50-25-10/sda_400Hz_80-50-25-10.cfg'
#                         '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_400Hz_100-50-25-10/sda_400Hz_100-50-25-10.cfg'
                        '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_100Hz_100-50-25-10/sda_100Hz_100-50-25-10.cfg'
                         );
#                          os.path.join(os.path.dirname(__file__), 'run', 'train_sda_mlp.cfg'), reset_logging=False);
#     config.dataset_suffix = '_channels';   
    config.dataset_root = '/Users/sstober/work/datasets/Dan/eeg';
    config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_400Hz_80-50-25-10/';
    config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_400Hz_100-50-25-10/';
    config.experiment_root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch_100Hz_100-50-25-10/';
#     analyze(config, subjects='all', worst_frames=True);
    analyze(config, subjects=[1], worst_frames=False);