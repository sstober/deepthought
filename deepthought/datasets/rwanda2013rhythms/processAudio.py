'''
Created on Mar 1, 2014

@author: sstober
'''

import os;


import wave;

import numpy as np;


import librosa;  # pip install librosa

import logging;
from config import Config;
    
from deepthought.util.fs_util import save;

def load():
    PATH = '/Users/sstober/work/datasets/Dan/audio/';
    filename = PATH+'180C_3.wav180F_4_180afr.wav';
    samplerate = 512;
    
    wave_file = wave.open(filename, 'r');
    
    
    print 'sample width: %d' % wave_file.getsampwidth();
    print 'num frames: %d' % wave_file.getnframes();
    print 'frame rate: %d Hz' % wave_file.getframerate();
    print 'channels: %d' % wave_file.getnchannels();
    print 'compression: %s' % wave_file.getcompname();
    print 'length: %f s' % (wave_file.getnframes() / wave_file.getframerate());
    
    samples = wave_file.readframes(wave_file.getnframes());
    
    print len(samples) / 4;
    
#     print samples;
        
    wave_file.close();
        
    loadfile(filename, samplerate);
    
    
def loadfile(filename, auto_sample_rate=False, samplerate=50, barsamples=100, maxbars=4):

    baselabel = os.path.basename(filename);

    # read beat length from file name
    beatlength = int(os.path.basename(filename)[0:3]);
    
    # 12 beats per bar, value in seconds
    barlength = beatlength * 0.012;

    # determine samplerate and samples per bar (barsamples)    
    if not auto_sample_rate:
        # a) fixed samplerate -> compute number of samples per bar (barsamples)
        barsamples = int(barlength * samplerate);
    else:
        # b) fixed barsamples -> compute number of samples per second (samplerate)
        samplerate = int(barsamples / barlength);
            
    y, sr = librosa.load(filename,  mono=True, sr=samplerate, dtype=np.float32) # offset=0.0, duration=None, dtype=<type 'numpy.float32'>);
    
    logging.info('loaded {0} : {1:d} samples for {2:.2f}s @ {3}Hz'.format(filename, len(y), len(y) / sr, sr));
    logging.debug('beat length = {0}ms  bar length = {1:.2f}s  samples per bar = {2}'.format(beatlength, barlength, barsamples)); 
    
    # chop after 2x2 bars = 2 repetitions
    y = y[0:maxbars*barsamples];
    
    # chop into 1 window per bar with no overlap
    frames = librosa.util.frame(y, frame_length=barsamples, hop_length=barsamples);

    # normalize to max amplitude 1
    frames = librosa.util.normalize(frames);
    # normalize to [0,1] with 0 -> 0.5 for usage with sigmoid units
    # not necessary if using tanh units
#     frames = 0.5 + frames / 2;

    # transpose to match Theano training sample format
    frames = np.transpose(frames);
    
    # convert to float32
    frames = np.asfarray(frames, dtype='float32');
#     print frames.dtype;

    labels = [];
    for i in range(0, len(frames)):
        labels.append(baselabel+'.'+str(i));
    labels = np.vstack(labels);

    logging.debug('{0} values'.format(frames.shape));
    
    return (frames, labels);

def loadall(path):
    logging.info('loading audio data from {0}'.format(path));
    data = []; # list
    labels = []; # list
    files = librosa.util.find_files(path, ext='wav', recurse=True); #, case_sensitive=False, limit=None, offset=0)
    for filename in files:
        x, y = loadfile(
                         filename, 
                         auto_sample_rate=config.audio.autpsamplerate, 
                         samplerate=config.audio.samplerate, 
                         barsamples=config.audio.barsamples,
                         maxbars=config.audio.maxbars);
        data.append(x);
        labels.append(y);
    data = np.vstack(data);  # transform list to a big numpy array by stacking
    labels = np.vstack(labels);
      
    logging.info('loaded {0} values from {1} files in total'.format(data.shape, len(files)));
#     print labels;
    return (data, labels);


if __name__ == '__main__':
    #pass
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG);
#     global config; 
    config = Config(file('deepbeat.cfg'));        
    logging.basicConfig(format=config.logger.pattern, level=logging.DEBUG);
    
    
    dataset = loadall(config.audio.path);
    #split = splitdata(dataset, ptest=config.audio.ptest, pvalid=config.audio.pvalid);
    #save(config.audio.datasetpath, split);
    save(config.audio.datasetpath, dataset);
#     load();