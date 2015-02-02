__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)


from pylearn2.utils.timing import log_timing
from pylearn2.train_extensions import TrainExtension
from pylearn2.space import CompositeSpace
import theano

from sklearn.metrics import confusion_matrix, classification_report

import numpy as np

class RNNMonitor(TrainExtension):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.__dict__.update(locals())

    def setup(self, model, dataset, algorithm):

        self.data_specs = (CompositeSpace((
                            model.get_input_space(),
                            model.get_output_space())),
                       ("features", "targets"))

        minibatch = self.model.get_input_space().make_theano_batch()
        # self.activation_fn = theano.function(
        #     inputs=[minibatch], outputs=self.model.fprop(minibatch, return_all=True))
        self.output_fn = theano.function(
            inputs=[minibatch], outputs=self.model.fprop(minibatch))


    def on_monitor(self, model, dataset, algorithm):

        # it = self.dataset.iterator('sequential', batch_size=1, data_specs=self.data_specs)

        # y_real, y_pred, output = process_dataset(self.model,
        #                                  self.dataset,
        #                                  data_specs=self.data_specs,
        #                                  output_fn=self.output_fn,
        #                                  batch_size=128)


        it = dataset.iterator(mode='sequential',
                              batch_size=128,
                              data_specs=self.data_specs)
        y_pred = []
        y_real = []
        output = []
        for minibatch, target in it:

            # note: axis 0 and 1 are swapped
            # frame_size, *, n_classes -> *, frame_size, n_classes
            target = target.swapaxes(0,1)
            out = self.output_fn(minibatch).swapaxes(0,1)
            output.append(out)
            # print out
            # print out.shape
            # print target.shape
            y_pred.append(np.argmax(out, axis = 2))
            y_real.append(np.argmax(target, axis = 2))

        # print output[-1].shape
        # print y_pred[-1].shape
        # print y_real[-1].shape

        y_pred = np.vstack(y_pred)
        # print y_pred.shape
        y_real = np.vstack(y_real)
        # print y_real.shape
        output = np.vstack(output)

        y_pred = y_pred.flatten()
        y_real = y_real.flatten()

        # Compute confusion matrix
        # print classification_report(y_real, y_pred)
        cm = confusion_matrix(y_real, y_pred)
        log.info('confusion\n{}'.format(cm))

        print classification_report(y_real, y_pred)

class DataDumper(TrainExtension):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.__dict__.update(locals())

    def setup(self, model, dataset, algorithm):

        # self.data_specs = (CompositeSpace((
        #                     model.get_input_space(),
        #                     model.get_output_space())),
        #                ("features", "targets"))

        self.data_specs = self.dataset.get_data_specs()
        print self.data_specs

        it = dataset.iterator(mode='sequential',
                              batch_size=1,
                              return_tuple=True,
                              data_specs=self.data_specs)

        i = 0
        for sequence, target in it:
            print '{}: {} -> {}'.format(i, sequence, target)
            # print np.asarray(sequence, dtype=np.int)
            # print np.asarray(target, dtype=np.int)
            i += 1

            break

        minibatch = self.model.get_input_space().make_theano_batch()
        # self.activation_fn = theano.function(
        #     inputs=[minibatch], outputs=self.model.fprop(minibatch, return_all=True))
        self.output_fn = theano.function(
            inputs=[minibatch], outputs=self.model.fprop(minibatch))


    def on_monitor(self, model, dataset, algorithm):

        it = dataset.iterator(mode='sequential',
                              batch_size=1,
                              return_tuple=True, # otherwise: "too many value to unpack"
                              data_specs=self.data_specs)

        i = 0
        for minibatch, target in it:

            # note: axis 0 and 1 are swapped
            # frame_size, *, n_classes -> *, frame_size, n_classes
            target = target.swapaxes(0,1)
            out = self.output_fn(minibatch).swapaxes(0,1)

            print '{}: {} ->\n{}\t expected {}'.format(i, minibatch, out, target)

            # print np.asarray(target, dtype=np.int)
            # print out.shape

            i += 1

            if i > 5: break

