'''
Created on Apr 7, 2014

@author: sstober
'''

import numpy as np;

audio_files = [
       '180C_10.wav180F_8_180afr.wav',
       '180C_10.wav180F_8_180west.wav',
       '180C_10.wav180F_9_180afr.wav',
       '180C_10.wav180F_9_180west.wav',
       '180C_11.wav180F_12_180west.wav',
       '180C_11.wav180F_13_180west.wav',
       '180C_12.wav180F_11_180west.wav',
       '180C_12.wav180F_13_180west.wav',
       '180C_13.wav180F_11_180west.wav',
       '180C_13.wav180F_12_180west.wav',
       '180C_3.wav180F_4_180afr.wav',
       '180C_3.wav180F_5_180afr.wav',
       '180C_4.wav180F_3_180afr.wav',
       '180C_4.wav180F_5_180afr.wav',
       '180C_5.wav180F_3_180afr.wav',
       '180C_5.wav180F_4_180afr.wav',
       '180C_8.wav180F_10_180afr.wav',
       '180C_8.wav180F_10_180west.wav',
       '180C_8.wav180F_9_180afr.wav',
       '180C_8.wav180F_9_180west.wav',
       '180C_9.wav180F_10_180afr.wav',
       '180C_9.wav180F_10_180west.wav',
       '180C_9.wav180F_8_180afr.wav',
       '180C_9.wav180F_8_180west.wav',
       '240C_10.wav240F_8_240afr.wav',
       '240C_10.wav240F_8_240west.wav',
       '240C_10.wav240F_9_240afr.wav',
       '240C_10.wav240F_9_240west.wav',
       '240C_11.wav240F_12_240west.wav',
       '240C_11.wav240F_13_240west.wav',
       '240C_12.wav240F_11_240west.wav',
       '240C_12.wav240F_13_240west.wav',
       '240C_13.wav240F_11_240west.wav',
       '240C_13.wav240F_12_240west.wav',
       '240C_3.wav240F_4_240afr.wav',
       '240C_3.wav240F_5_240afr.wav',
       '240C_4.wav240F_3_240afr.wav',
       '240C_4.wav240F_5_240afr.wav',
       '240C_5.wav240F_3_240afr.wav',
       '240C_5.wav240F_4_240afr.wav',
       '240C_8.wav240F_10_240afr.wav',
       '240C_8.wav240F_10_240west.wav',
       '240C_8.wav240F_9_240afr.wav',
       '240C_8.wav240F_9_240west.wav',
       '240C_9.wav240F_10_240afr.wav',
       '240C_9.wav240F_10_240west.wav',
       '240C_9.wav240F_8_240afr.wav',
       '240C_9.wav240F_8_240west.wav'
       ]

short_labels = [
       '10_8a',
       '10_8w',
       '10_9a',
       '10_9w',
       '11_12w',
       '11_13w',
       '12_11w',
       '12_13w',
       '13_11w',
       '13_12w',
       '3_4a',
       '3_5a',
       '4_3a',
       '4_5a',
       '5_3a',
       '5_4a',
       '8_10a',
       '8_10w',
       '8_9a',
       '8_9w',
       '9_10a',
       '9_10w',
       '9_8a',
       '9_8w',
                ]

meta_labels = [
               [ 6,  4, 'a'],
               [ 3,  1, 'w'],
               [ 6,  5, 'a'],
               [ 3,  2, 'w'],
               [ 4,  5, 'w'],
               [ 4,  6, 'w'],
               
               [ 5,  4, 'w'],
               [ 5,  6, 'w'],
               [ 6,  4, 'w'],
               [ 6,  5, 'w'],               
               [ 1,  2, 'a'],
               [ 1,  3, 'a'],
               
               [ 2,  1, 'a'],
               [ 2,  3, 'a'],
               [ 3,  1, 'a'],
               [ 3,  2, 'a'],
               [ 4,  6, 'a'],               
               [ 1,  3, 'w'],
               
               [ 4,  5, 'a'],
               [ 1,  2, 'w'],
               [ 5,  6, 'a'],
               [ 2,  3, 'w'],
               [ 5,  4, 'a'],
               [ 2,  1, 'w']
               ]

# swapped_meta_labels = [
#                [ 1,  2, 'a'],
#                [ 2,  1, 'a'],
#                [ 1,  3, 'a'],
#                [ 3,  1, 'a'],
#                [ 2,  3, 'a'],
#                [ 3,  2, 'a'],
# 
#                [ 4,  5, 'a'],
#                [ 5,  4, 'a'],
#                [ 4,  6, 'a'],
#                [ 6,  4, 'a'],               
#                [ 5,  6, 'a'],
#                [ 6,  5, 'a'],
#                
#                [ 1,  2, 'w'],
#                [ 2,  1, 'w'],
#                [ 1,  3, 'w'],
#                [ 3,  1, 'w'],
#                [ 2,  3, 'w'],
#                [ 3,  2, 'w'],
# 
#                [ 4,  5, 'w'],
#                [ 5,  4, 'w'],
#                [ 4,  6, 'w'],
#                [ 6,  4, 'w'],
#                [ 5,  6, 'w'],
#                [ 6,  5, 'w'],
#                ]

swapped_meta_labels = [
               [ 1,  2, 'a'],
               [ 1,  3, 'a'],
               [ 2,  1, 'a'],
               [ 2,  3, 'a'],
               [ 3,  1, 'a'],
               [ 3,  2, 'a'],

               [ 4,  5, 'a'],
               [ 4,  6, 'a'],
               [ 5,  4, 'a'],
               [ 5,  6, 'a'],
               [ 6,  4, 'a'],               
               [ 6,  5, 'a'],
               
               [ 1,  2, 'w'],
               [ 1,  3, 'w'],
               [ 2,  1, 'w'],
               [ 2,  3, 'w'],
               [ 3,  1, 'w'],
               [ 3,  2, 'w'],

               [ 4,  5, 'w'],
               [ 4,  6, 'w'],
               [ 5,  4, 'w'],
               [ 5,  6, 'w'],
               [ 6,  4, 'w'],
               [ 6,  5, 'w'],
               ]


class LabelConverter(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.shuffle_classes = np.zeros(48, dtype=np.int8);
        
        self.class_labels = {};
        self.class_labels['rhythm'] = audio_files;
        self.class_labels['rhythm_type'] = ['African', 'Western'];
        self.class_labels['tempo'] = ['180', '240'];
        
        self.stimulus_id_map = {};
        self.label_map = {}; #np.empty(len(audio_files));
        for i, audio_file in enumerate(audio_files):
            labels = {};
            labels['audio_file'] = audio_file;
            
            if '180' in audio_file:
#                 labels['tempo'] = 180;
                labels['tempo'] = 0;
            else:
#                 labels['tempo'] = 240;
                labels['tempo'] = 1;
        
            if 'west' in audio_file:
#                 labels['rhythm_type'] = 'W';
#                 labels['rhythm_type'] = 1;
                labels['rhythm_type'] = self.class_labels['rhythm_type'].index('Western');
            else:
#                 labels['rhythm_type'] = 'A';
#                 labels['rhythm_type'] = 0;
                labels['rhythm_type'] = self.class_labels['rhythm_type'].index('African');
            
            rhythm_meta_labels = meta_labels[i % 24]; # map down to 24 classes
            mapped_rhythm_id = swapped_meta_labels.index(rhythm_meta_labels);
            self.shuffle_classes[i] = mapped_rhythm_id;

            labels['rhythm_meta'] = rhythm_meta_labels
            
#             labels['rhythm'] = mapped_rhythm_id; #'{}{}{}'.format(*rhythm_meta_labels);             
#             self.stimulus_id_map[audio_file] = mapped_rhythm_id; # i;        
#             self.label_map[mapped_rhythm_id] = labels;
            
            labels['rhythm'] = i % 24; # map down to 24 classes
            self.stimulus_id_map[audio_file] = i;        
            self.label_map[i] = labels;
        
    def get_class_labels(self, label_mode):
        return self.class_labels[label_mode];
        
    def get_stimulus_id(self, stimulus):
        return self.stimulus_id_map[stimulus];
    
    def get_tempo_label(self, stimulus_id):
        return self.label_map[stimulus_id]['tempo'];
        
    def get_rhythm_type_label(self, stimulus_id):
        return self.label_map[stimulus_id]['rhythm_type'];
    
    def get_audio_file(self, stimulus_id):
        return self.label_map[stimulus_id]['audio_file'];
    
    def get_label(self, stimulus_id, label_mode):
        return self.label_map[stimulus_id][label_mode];
    
    def get_labels(self, stimulus_ids, label_mode):
        labels = [];
#         counts = np.zeros(50);
        for i in xrange(len(stimulus_ids)):
            stimulus_id = stimulus_ids[i][0]; # FIXME 
#             counts[stimulus_id] += 1;
            labels.append(self.label_map[stimulus_id][label_mode]);
#         print labels.count(180);
#         print labels.count(240);
#         print counts;
#         return np.vstack(labels);
        return labels;