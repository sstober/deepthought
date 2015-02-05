'''
Created on Jun 20, 2014

@author: sstober
'''

import random;
import collections;
import math;

class SimpleDatasetPartitioner(object):
    '''
    classdocs
    '''


    def __init__(self, p_valid, p_test, random_seed):
        '''
        Constructor
        '''
        self.p_valid = p_valid;
        self.p_test = p_test;
        self.random_seed = random_seed;
        
    def get_partition(self, name, metadb):
        
        files_per_label = collections.defaultdict(list);
        for datafile, metadata in metadb.items():
            files_per_label[metadata['label']].append(datafile);
            
        selected = [];
        for label, datafiles in files_per_label.items():
            print '{}: {}'.format(label, datafiles);
        
            permutation = sorted(datafiles[:]);
            random.seed(self.random_seed);
            random.shuffle(permutation);
            
            n_valid = int(math.ceil(len(datafiles) / 100. * self.p_valid));
            n_test = int(math.ceil(len(datafiles) / 100. * self.p_test));
            
            print 'p_valid: {}\tn_valid: {}'.format(self.p_valid, n_valid);
            print 'p_test:  {}\tn_test:  {}'.format(self.p_test, n_test);
            
            if name == 'valid':
                selected.extend(datafiles[0:n_valid]);
            elif name == 'test':
                selected.extend(datafiles[n_valid:n_valid+n_test]);
            elif name == 'train':
                selected.extend(datafiles[n_valid+n_test:]);
        
        print 'selected for {}: {}'.format(name, selected);
        
        return selected;
        