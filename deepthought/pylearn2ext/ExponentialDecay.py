'''
Created on Apr 30, 2014

@author: sstober
'''

import numpy as np;
import theano;
from pylearn2.train_extensions import TrainExtension;

class ExponentialDecay(TrainExtension):

    def __init__(self, decay_factor, min_lr):
        if isinstance(decay_factor, str):
            decay_factor = float(decay_factor)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        assert isinstance(decay_factor, float)
        assert isinstance(min_lr, float)
        self.__dict__.update(locals())
        del self.self
        self._count = 0
        self._min_reached = False

    def on_monitor(self, model, dataset, algorithm):
        """
        Updates the learning rate according to the exponential decay schedule.

        Parameters
        ----------
        algorithm : SGD
            The SGD instance whose `learning_rate` field should be modified.
        """
        if self._count == 0:
            self._base_lr = algorithm.learning_rate.get_value()
        self._count += 1

        if not self._min_reached:
            # If we keep on executing the exponentiation on each mini-batch,
            # we will eventually get an OverflowError. So make sure we
            # only do the computation until min_lr is reached.
            new_lr = self._base_lr / (self.decay_factor ** self._count)
            if new_lr <= self.min_lr:
                self._min_reached = True
                new_lr = self.min_lr
        else:
            new_lr = self.min_lr

        new_lr = np.cast[theano.config.floatX](new_lr)
        algorithm.learning_rate.set_value(new_lr)
            