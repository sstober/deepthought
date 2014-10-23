'''
Created on Apr 23, 2014

@author: sstober
'''

from pylearn2.models.mlp import RectifiedLinear;
from pylearn2.utils import wraps

class PretrainedRLU(RectifiedLinear):
    '''
    classdocs
    '''
    def __init__(self, layer_content=None, **kwargs):
        
        assert layer_content is not None;
        dim = layer_content.nhid;
        self.layer_content = layer_content; # needed later
        
        super(PretrainedRLU, self).__init__(dim=dim, **kwargs)
                
        self.set_biases(layer_content.hidbias.get_value());
    
    @wraps(RectifiedLinear.set_input_space)
    def set_input_space(self, space):
        self.irange = 0; # hack tp avoid assert error; value does not matter
        super(PretrainedRLU, self).set_input_space(space);
        self.set_weights(self.layer_content.weights.get_value());