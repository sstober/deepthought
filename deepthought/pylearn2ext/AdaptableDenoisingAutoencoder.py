'''
Created on Apr 3, 2014

@author: sstober
'''

import logging;
log = logging.getLogger(__name__);

import theano;
from pylearn2.models.autoencoder import DenoisingAutoencoder;

class AdaptableDenoisingAutoencoder(DenoisingAutoencoder):
    def __init__(self, corruptor, nvis, nhid, act_enc, act_dec,
                 tied_weights=False, irange=1e-3, rng=9001, name=''):
        super(AdaptableDenoisingAutoencoder, self).__init__(
            corruptor,
            nvis,
            nhid,
            act_enc,
            act_dec,
            tied_weights,
            irange,
            rng
        );
    def get_data(self):
        return self.X;
    
    def untie_weights(self):
        if not self.tied_weights:
            raise ValueError("%s is not a tied-weights autoencoder" % str(self));
        self.w_prime = theano.shared(self.weights.get_value(borrow=False).T,
                                     name='w_prime')
        self.tied_weights = False;
        self._params.append(self.w_prime);
#         self.redo_theano(); # maybe this helps

    def tie_weights(self):
        if self.tied_weights:
            raise ValueError("%s is not a untied-weights autoencoder" % str(self));
        self._params.remove(self.w_prime);
        self.w_prime = self.weights.T;
        self.tied_weights = True;
        
    def set_corruption_level(self, corruption):
#         self.curruptor = AdaptableCorruptor(corruption_level=corruption, name=self.corruptor.name);
        if self.corruptor.corruption_level != corruption:
            log.warn('changing corruption level from {} to {}. may need to reset function graph.'.format(
                                self.corruptor.corruption_level, corruption));
            self.corruptor.corruption_level = corruption; 
            self.redo_theano(); # FIXME: maybe this helps
        else:
            log.debug('corruption level unchanged at {}'.format(self.corruptor.corruption_level));
        
    def __str__(self):
        s = 'AdaptableDenoisingAutoencoder';
        if (self.name):
            s += ' '+self.name;
        return s;