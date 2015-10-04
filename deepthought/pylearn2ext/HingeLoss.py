'''
Created on Apr 30, 2014

@author: 
original code by Kyle Kastner from
https://github.com/kastnerkyle/pylearn2/blob/svm_layer/pylearn2/models/mlp.py
adapted by Sebastian Stober according to recent API changes in pylearn2
'''

import numpy as np;
from theano import config;
from theano.gof.op import get_debug_values
from theano.printing import Print
import theano.tensor as T;
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import Layer;
from pylearn2.space import Space, VectorSpace, Conv2DSpace;
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
import warnings;

class HingeLoss(Layer):

    def __init__(self, n_classes, layer_name, irange = None,
                 istdev = None,
                 no_affine=False,
                 sparse_init = None):
        
        super(HingeLoss, self).__init__();

        self.__dict__.update(locals())
        del self.self

        self.output_space = VectorSpace(n_classes)

        if not self.no_affine:
            self.b = sharedX(np.zeros((n_classes,)), name = 'hingeloss_b')

    def get_monitoring_channels(self):

        if self.no_affine:
            return OrderedDict()

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

        # channels that does not require state information
#         if self.no_affine:
#             rval = OrderedDict()
#
#         W = self.W
# 
#         assert W.ndim == 2
# 
#         sq_W = T.sqr(W)
# 
#         row_norms = T.sqrt(sq_W.sum(axis=1))
#         col_norms = T.sqrt(sq_W.sum(axis=0))
# 
#         rval = OrderedDict([('row_norms_min',  row_norms.min()),
#                             ('row_norms_mean', row_norms.mean()),
#                             ('row_norms_max',  row_norms.max()),
#                             ('col_norms_min',  col_norms.min()),
#                             ('col_norms_mean', col_norms.mean()),
#                             ('col_norms_max',  col_norms.max()), ])

        rval = OrderedDict()
        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=1)

            rval.update(OrderedDict([
                                ('mean_max_class', mx.mean()),
                                ('max_max_class', mx.max()),
                                ('min_max_class', mx.min())]))

            if targets is not None:
                y_hat = self.target_convert(T.argmax(state, axis=1))
                #Assume target is in [0,1] as binary one-hot
                y = self.target_convert(T.argmax(targets, axis=1))
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
                rval['misclass'] = misclass
                rval['nll'] = self.cost(Y_hat=state, Y=targets)

        return rval


    def get_monitoring_channels_from_state(self, state, target=None):
        warnings.warn("Layer.get_monitoring_channels_from_state is " + \
                    "deprecated. Use get_layer_monitoring_channels " + \
                    "instead. Layer.get_monitoring_channels_from_state " + \
                    "will be removed on or after september 24th 2014",
                    stacklevel=2)

        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            y_hat = self.target_convert(T.argmax(state, axis=1))
            #Assume target is in [0,1] as binary one-hot
            y = self.target_convert(T.argmax(target, axis=1))
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)

        return rval

    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.no_affine:
            self._params = []
        else:
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_classes))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W = rng.randn(self.input_dim, self.n_classes) * self.istdev
            else:
                assert self.sparse_init is not None
                W = np.zeros((self.input_dim, self.n_classes))
                for i in xrange(self.n_classes):
                    for j in xrange(self.sparse_init):
                        idx = rng.randint(0, self.input_dim)
                        while W[idx, i] != 0.:
                            idx = rng.randint(0, self.input_dim)
                        W[idx, i] = rng.randn()

            self.W = sharedX(W,  'hingeloss_W' )

            self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.np_format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt,
                                         self.input_space.axes,
                                         ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):
        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            rval = state_below
        else:
            assert self.W.ndim == 2
            b = self.b
            W = self.W

            rval = T.dot(state_below, W) + b

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def target_convert(self, Y):
        '''
        converts target [0,1] to [-1, 1]
        '''
        Y_t = 2. * Y - 1.
        return Y_t

    # def hinge_cost(self, W, Y, Y_hat, C=1.):
    def hinge_cost(self, Y, Y_hat):
        #prob = .5 * T.dot(self.W.T, self.W) + C * (T.maximum(1 - Y * Y_hat, 0) ** 2.).sum(axis=1)
        prob = (T.maximum(1 - Y * Y_hat, 0) ** 2.).sum(axis=1)
        return prob

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a hinge loss estimate.
        of Y.
        """

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert Y_hat.ndim == 2
        Y_t = self.target_convert(Y)
        # prob = self.hinge_cost(self.W, Y_t, Y_hat)
        prob = self.hinge_cost(Y_t, Y_hat)
        assert prob.ndim == 1
        rval = prob.mean()

        return rval


    def cost_matrix(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a hinge loss estimate.
        of Y.
        """

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op

        assert Y_hat.ndim == 2
        Y_t = self.target_convert(Y)
        # prob = self.hinge_cost(self.W, Y_t, Y_hat)
        prob = self.hinge_cost(Y_t, Y_hat)
        return prob


    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        if self.no_affine:
            return