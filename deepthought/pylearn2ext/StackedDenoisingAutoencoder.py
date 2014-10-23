'''
Created on Apr 3, 2014

@author: sstober
'''

import logging;
log = logging.getLogger(__name__);

from pylearn2.models.autoencoder import DeepComposedAutoencoder;

class StackedDenoisingAutoencoder(DeepComposedAutoencoder):
    
    def __init__(self, autoencoders, corruptor):    
        self.corruptor = corruptor;
        super(StackedDenoisingAutoencoder, self).__init__(autoencoders);
    
    def reconstruct(self, inputs):
        corrupted = self.corruptor(inputs)
        return super(StackedDenoisingAutoencoder, self).reconstruct(corrupted)
    
    def output(self, inputs):
        return self.decode(self.encode(inputs));
    