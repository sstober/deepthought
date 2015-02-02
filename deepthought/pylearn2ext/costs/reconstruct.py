__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost

import theano.tensor as T

from deepthought.util.axes_util import symbolic_to_b01c


class MeanSquaredReconstructionError(DefaultDataSpecsMixin, Cost):
    """
    for each input instance,
    """

    @staticmethod
    def cost(a, b):
        """
        .. todo::

            WRITEME
        """
        l = a.shape[1] * a.shape[2] * a.shape[3]
        a = T.reshape(a, (a.shape[0], l))
        b = T.reshape(b, (b.shape[0], l))
        return ((a - b) ** 2).sum(axis=1).mean()

    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X = data

        X_out = T.swapaxes(T.swapaxes(model.fprop(X), 1, 2), 2,3)  # convert bc01 to b01c
        return self.cost(X, X_out)

class LayerMeanSquaredReconstructionError(DefaultDataSpecsMixin, Cost):
    """
    for each input instance,
    """
    def __init__(self,
                 model,
                 layer1_name,
                 layer2_name,
                 layer1_axes='b01c',
                 layer2_axes='b01c'):

        # determine layer ids
        def get_layer_id(model, layer_name):
            for i, layer in enumerate(model.layers):
                if layer.layer_name == layer_name:
                    return i
            raise ValueError('Layer not found: {}'.format(layer_name))

        self.layer1_id = get_layer_id(model, layer1_name)
        self.layer2_id = get_layer_id(model, layer2_name)

        self.layer1_axes = layer1_axes
        self.layer2_axes = layer2_axes


    @staticmethod
    def cost(a, b):
        """
        .. todo::

            WRITEME
        """
        l = a.shape[1] * a.shape[2] * a.shape[3]
        a = T.reshape(a, (a.shape[0], l))
        b = T.reshape(b, (b.shape[0], l))
        return ((a - b) ** 2).sum(axis=1).mean()

    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)

        act = model.fprop(data, return_all=True)

        # get layer activations and convert to b01c
        act1 = symbolic_to_b01c(act[self.layer1_id], self.layer1_axes)
        act2 = symbolic_to_b01c(act[self.layer2_id], self.layer2_axes)

        # act1 = theano.printing.Print('act1: ', attrs=('shape',))(act1)
        # act2 = theano.printing.Print('act2: ', attrs=('shape',))(act2)

        return self.cost(act1, act2)

from theano.printing import Print
import pylearn2.costs.dbm

class TargetMeanSquaredReconstructionError(DefaultDataSpecsMixin, Cost):
    """
    for each input instance,
    """
    def __init__(self):
        self.supervised = True # for DefaultDataSpecsMixin

    @staticmethod
    def cost(a, b):
        """
        .. todo::

            WRITEME
        """
        return ((a - b) ** 2).sum(axis=2).mean()

    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        # print self.get_data_specs(model)
        X, Y = data
        assert Y is not None

        # X = Print()

        X_out = model.fprop(X)
        # return self.cost(Y, X_out)
        return self.cost(Y, X_out[-1]) # FIXME: this is a temporary hack for HAMR