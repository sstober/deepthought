'''
Created on Apr 22, 2014

@author: sstober
'''

import logging;

log = logging.getLogger(__name__);

import os;
import numpy as np
from deepthought.experiments.ismir2014.util import load_config;

import matplotlib.pyplot as plt
from pylearn2.utils import serial

from deepthought.experiments.ismir2014.extract_results import compare_best_result;

# from pylearn2.scripts.plot_monitor import plot_monitor;

import colorsys

def get_colors(i, total):
    hue = i*(1.0/total)
    dark = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 100)]
    light = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
    return "#{0:02x}{1:02x}{2:02x}".format(*dark), "#{0:02x}{1:02x}{2:02x}".format(*light)

def get_dark_light_colors(total):
    dark = [];
    light = [];
    for i in range(total):
        hue = i*(1.0/total)
        d = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 100)]
        l = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        dark.append("#{0:02x}{1:02x}{2:02x}".format(*d));
        light.append("#{0:02x}{1:02x}{2:02x}".format(*l));
    return dark + light;

def get_color_variants(total):
    dark = [];
    middle = [];
    light = [];
    for i in range(total):
        hue = i*(1.0/total)
        d = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        m = [int(x) for x in colorsys.hsv_to_rgb(hue, 0.40, 230)]
        l = [int(x) for x in colorsys.hsv_to_rgb(hue, 0.20, 230)]
        dark.append("#{0:02x}{1:02x}{2:02x}".format(*d));
        middle.append("#{0:02x}{1:02x}{2:02x}".format(*m));
        light.append("#{0:02x}{1:02x}{2:02x}".format(*l));
    return dark + middle + light;


def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)

mode = 'w';
channels = [
            'test_'+mode+'trial_misclass_rate',
            'test_'+mode+'seq_misclass_rate',
            'test_y_misclass',
            'test_y_nll',
            
            'valid_'+mode+'trial_misclass_rate',
            'valid_'+mode+'seq_misclass_rate',
            'valid_y_misclass',
            'valid_y_nll',
            
            'train_'+mode+'trial_misclass_rate',
            'train_'+mode+'seq_misclass_rate',
            'train_y_misclass',
            'train_y_nll',
            
            'train_objective',
            'learning_rate',
            'momentum'
            ];

less_channels = [
#             'test_ptrial_misclass_rate',
#             'test_pseq_misclass_rate',
            'test_y_misclass',
            'test_y_nll',
            
#             'valid_ptrial_misclass_rate',
#             'valid_pseq_misclass_rate',
            'valid_y_misclass',
            'valid_y_nll',
            
#             'train_ptrial_misclass_rate',
#             'train_pseq_misclass_rate',
            'train_y_misclass',
            'train_y_nll',
            
            'train_objective',
            'learning_rate',
            'momentum'
            ];
            
less_colors = get_color_variants(2);
            
def get_xy(channel, channel_name='', x_axis='epoch'):
    y = np.asarray(channel.val_record)        

    if np.any(np.isnan(y)):
        print channel_name + ' contains NaNs'

    if np.any(np.isinf(y)):
        print channel_name + 'contains infinite values'

    if x_axis == 'example':
        x = np.asarray(channel.example_record)
    elif x_axis == 'batche':
        x = np.asarray(channel.batch_record)
    elif x_axis == 'epoch':
        try:
            x = np.asarray(channel.epoch_record)
        except AttributeError:
            # older saved monitors won't have epoch_record
            x = np.arange(len(channel.batch_record))
    elif x_axis == 'second':
        x = np.asarray(channel.time_record)
    elif x_axis == 'hour':
        x = np.asarray(channel.time_record) / 3600.
    else:
        assert False
    
    return x,y;

def plot(model_path, channel_names, file_path=None, colors=None):
    
    print model_path;
    
    model = serial.load(model_path);
    
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     colors = ['b', 'g', 'r', 'c']

    if colors == None:
        colors = get_color_variants(4);
    
    styles = list(colors)
#     styles += [color+'--' for color in colors]
#     styles += [color+':' for color in colors]

    fig = plt.figure(dpi=300)
    ax = plt.subplot(1,1,1)
    plt.ylim(0,1);
    
    ax2 = ax.twinx();
    ax2.set_yscale('log')
    
    ax3 = ax.twinx();
    plt.ylim(0,2);
    
    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.80)
    
    # Move the last y-axis spine over to the right by 20% of the width of the axes
    ax3.spines['right'].set_position(('axes', 1.10))
    
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    
    channels = model.monitor.channels
#     print channels;
    
    x_axis='epoch'
    for idx, channel_name in enumerate(channel_names):
        if not channel_name in channels:
            print channel_name+' missing';
            continue;
        
        x,y = get_xy(channels[channel_name], channel_name, x_axis);
            
        if channel_name == 'learning_rate':
            ax2.plot( x,
                  y,
                  'k',
#                   styles[idx % len(styles)],
#                   marker = '.', # add point margers to lines
                  label = channel_name)
        elif channel_name in ['momentum']:
            ax.plot( x,
                  y,
                  'k',
                  label = channel_name)
        elif channel_name in ['train_objective']:
            color = 'g';
            ax3.plot( x,
                  y,
                  color,
#                   label = channel_name
                  )
            ax3.set_ylabel('train_objective', color=color)
            ax3.tick_params(axis='y', colors=color)
        else:
            ax.plot( x,
                  y,
                  styles[idx % len(styles)],
                  linewidth=0.9,
#                   marker = '.', # add point margers to lines
                  label = channel_name)
        
    plt.xlabel('# '+x_axis+'s')
    ax.ticklabel_format( scilimits = (-3,3), axis = 'both')

#     handles, labels = ax.get_legend_handles_labels()
#     lgd = ax.legend(handles, labels, loc='upper center',
#             bbox_to_anchor=(0.5,-0.1))
    # 0.046 is the size of 1 legend box
#     fig.subplots_adjust(bottom=0.11 + 0.046 * len(channel_names))
#     fig.subplots_adjust(bottom= 0.046 * len(channel_names))

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, dpi=300)    


def plot_batch(batch_root):
    for i in xrange(13):
        experiment_root = os.path.join(batch_root, 'subj'+str(i+1));
        model_path = os.path.join(experiment_root, 'mlp.pkl');
                 
        plot(model_path, channels, experiment_root+'/plot.pdf');
        plot(model_path, less_channels, experiment_root+'/plot-less.pdf', get_color_variants(2));
        
def scan_for_best_performance(experiment_root, scan_channel='valid_y_misclass'):
    model = serial.load(os.path.join(experiment_root, 'mlp.pkl'));
    channels = model.monitor.channels;
    
    best_epoch = np.argmin(channels[scan_channel].val_record);
    print 'best epoch for channel {}: {}'.format(scan_channel, best_epoch);
    
    selected_channels = [
            'test_ptrial_misclass_rate',
            'test_wtrial_misclass_rate',
            'test_trial_misclass_rate',
            'test_pseq_misclass_rate',
            'test_wseq_misclass_rate',
            'test_seq_misclass_rate',
            'test_y_misclass',
#             'test_y_nll',
            
            'valid_ptrial_misclass_rate',
            'valid_wtrial_misclass_rate',
            'valid_trial_misclass_rate',
            'valid_pseq_misclass_rate',
            'valid_wseq_misclass_rate',
            'valid_seq_misclass_rate',
            'valid_y_misclass',
#             'valid_y_nll',
            
            'train_wtrial_misclass_rate',
            'train_trial_misclass_rate',
            'train_wseq_misclass_rate',
            'train_seq_misclass_rate',
            'train_y_misclass',
#             'train_y_nll',      
            
            'post_wtrial_misclass_rate',
            'post_trial_misclass_rate',
            'post_wseq_misclass_rate',
            'post_seq_misclass_rate',
            'post_y_misclass',
#             'post_y_nll',        
            ];
    for channel in selected_channels:
        if not channel in channels:
            continue;
        
        print '{}: {:.3f}'.format(channel, float(channels[channel].val_record[-1]));
        
    

def print_best_performance(experiment_root):
    print '-------' + experiment_root + '-------';
    model = serial.load(os.path.join(experiment_root, 'mlp_best.pkl'));
    
    channels = model.monitor.channels;
    
    selected_channels = [
            'test_wtrial_misclass_rate',
            'test_wseq_misclass_rate',
            'test_y_misclass',
            'test_y_nll',
            
            'valid_wtrial_misclass_rate',
            'valid_wseq_misclass_rate',            
            'valid_y_misclass',
            'valid_y_nll',
            
            'train_wtrial_misclass_rate',
            'train_wseq_misclass_rate',
            'train_y_misclass',
            'train_y_nll',      
            
            'post_ptrial_misclass_rate',
            'post_wtrial_misclass_rate',
            'post_pseq_misclass_rate',
            'post_wseq_misclass_rate',
            'post_y_misclass',
            'post_y_nll',        
            ];
    
    for channel in selected_channels:
        if not channel in channels:
            continue;
        
        try:
            print '{}: {}'.format(channel, channels[channel].val_record[-1]);
        except:
            log.error('unable to plot channel {}'.format(channel));
            
def plot2(root):
    plot(root+'/mlp.pkl', channels, root+'/plot.pdf');   
    plot(root+'/mlp.pkl', less_channels, root+'/plot-less.pdf', get_color_variants(2));
    
def plot_all(root_path):
    import fnmatch    

#     matches = []
    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, 'mlp.pkl'):
#             matches.append(os.path.join(root, filename))
            plot2(root);

if __name__ == '__main__':
    config = load_config(default_config=
#                 os.path.join(os.path.dirname(__file__), '..', 'run', 'train_convnet.cfg'), reset_logging=True);
                os.path.join(os.path.dirname(__file__), '..', 'run', 'train_fftconvnet.cfg'), reset_logging=True);
#                 os.path.join(os.path.dirname(__file__), '..', 'run', 'train_sda_mlp.cfg'), reset_logging=True);
    
    # FIXME:
#     root = '/Users/stober/git/deepbeat/deepbeat/output/gpu/sda/exp6.14all/';
#     root = '/Users/stober/git/deepbeat/deepbeat/output/gpu/cnn/bigbatch/individual/';
#     plot_batch(root);
    
#     for i in xrange(12):
#         root = '/Users/stober/git/deepbeat/deepbeat/output/gpu/cnn/bigbatch/cross-trial/pair'+str(i);
#         plot(root+'/mlp.pkl', channels, root+'/plot.pdf');   
#         plot(root+'/mlp.pkl', less_channels, root+'/plot-less.pdf', get_color_variants(2));
    
    # FIXME
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/subj2_50-25-10.f';
#     root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/exp04.4b';
#     root = '/Users/sstober/git/deepbeat/deepbeat/output/sda-mlp4';
#     root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/cnn/exp4.1';
#     root = '/Users/stober/git/deepbeat/deepbeat/output/gpu/cnn/bigbatch/24subj2/';
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/cnn/exp08.2-bigbatch/subj2.13';

    plot2(root);
    raise ''
#     extract_results(root, config);
    
    # recursive stuff
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/hyper1.1';
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/hyper2b';
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/exp05.9';
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/sda/batch3';
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/cross-test2/'
    root = '/Users/sstober/git/deepbeat/deepbeat/output/gpu/fftcnn/exp08.a-crosstrial.subj9/'; 
    plot_all(root);
#     compare_best_result(root, mode='f1', check_dataset='valid');
    compare_best_result(root, mode='misclass', check_dataset='valid');
    
#     scan_for_best_performance(root);

#     scan_for_best_performance(root);
#     scan_for_best_performance(root, scan_channel='valid_wtrial_misclass_rate');
#     scan_for_best_performance(root, scan_channel='test_wtrial_misclass_rate');
