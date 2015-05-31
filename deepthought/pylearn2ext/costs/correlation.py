__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace

import theano.tensor as T

class MeanCrossCorrelation(DefaultDataSpecsMixin, Cost):

    def __init__(self, supervised=True):
        self.supervised = supervised


    def _get_desired_space(self, space):
        if isinstance(space, Conv2DSpace):
            return Conv2DSpace(shape=space.shape,
                                        num_channels=space.num_channels,
                                        axes=('b', 'c', 0, 1))
        elif isinstance(space, VectorSpace):
            return space
        else:
            raise ValueError('Space not supported: {}'.format(space))


    # def _get_data_specs(self, model):
    #     desired_space = self._get_desired_space(model.get_input_space())
    #
    #     if self.supervised:
    #         space = CompositeSpace([desired_space, desired_space])
    #         sources = (model.get_input_source(), model.get_target_source())
    #         return (space, sources)
    #     else:
    #         return (desired_space, model.get_input_source())


    @staticmethod
    def cost(a, b):
        """
        .. todo::

            WRITEME
        """
        # for debugging: check that batch shape is identical
        # import theano
        # a = theano.printing.Print('a: ', attrs=('shape',))(a)
        # b = theano.printing.Print('b: ', attrs=('shape',))(b)

        # nb = a.shape[0]
        # nc = a.shape[1]

        l0 = a.shape[0] * a.shape[1]
        l1 = a.shape[2] * a.shape[3]
        a = T.reshape(a, (l0, l1))
        b = T.reshape(b, (l0, l1))

        r = pairwise_pearsonr(a,b)

        # r = r.reshape((nb, nc))
        #
        # import theano
        # r = theano.printing.Print('r: ', attrs=('shape',))(r)
        #
        # r = r.mean(axis=1)
        # r = theano.printing.Print('r: ', attrs=('shape',))(r)

        return 1. - r.mean()


    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """

        if self.supervised:
            input_space = self.get_data_specs(model)[0].components[0] # data_specs[0] is CompositeSpace
            desired_space = self._get_desired_space(input_space)

            X, Y = data
            input_space.validate(X)

            target_space = model.get_target_space()
            target_space.validate(Y)
            Y = target_space.format_as(Y, desired_space)

            Y_out = model.fprop(X)
            output_space = model.get_output_space()
            output_space.validate(Y_out)
            Y_out = output_space.format_as(Y_out, desired_space)

            return self.cost(Y, Y_out)

        else:
            input_space = self.get_data_specs(model)[0]
            desired_space = self._get_desired_space(input_space)

            X = data
            input_space.validate(X)
            Y = model.fprop(X)

            X = input_space.format_as(X, desired_space) # needs to be done after fprop!

            output_space = model.get_output_space()
            output_space.validate(Y)
            Y = output_space.format_as(Y, desired_space)

            return self.cost(X, Y)


def pairwise_pearsonr(X,Y):
    # remove DC
    mx = X.mean(axis=1)
    my = Y.mean(axis=1)
    # mx = x.mean()
    # my = y.mean()
    # xm, ym = x-mx, y-my
    xm, ym = (X.T-mx).T, (Y.T-my).T
    # print xm.shape

    # r_num = np.add.reduce(xm * ym)
    r_num = T.sum(xm * ym, axis=1)

    # r_den = np.sqrt(ss(xm) * ss(ym))
    def _ss(a):
        return T.sum(a*a, axis=1)
    r_den = T.sqrt(_ss(xm) * _ss(ym))

    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = T.maximum(T.minimum(r, 1.0), -1.0)

    return r