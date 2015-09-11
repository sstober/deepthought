__author__ = 'sstober'

from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from copy import deepcopy
from pylearn2.utils import serial
from pylearn2.utils.timing import log_timing

import logging
log = logging.getLogger(__name__)


class MonitorBasedSaveBestMod(MonitorBasedSaveBest):

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier
        - or equally good (modification).
        If it's the case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            model.monitor must contain a channel with name given by
            self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if self.coeff * new_cost <= self.coeff * self.best_cost and \
           monitor._epochs_seen >= self.start_epoch:
            self.best_cost = new_cost
            # Update the tag of the model object before saving it.
            self._update_tag(model)
            if self.store_best_model:
                self.best_model = deepcopy(model)
            if self.save_path is not None:
                with log_timing(log, 'Saving to ' + self.save_path):
                    serial.save(self.save_path, model, on_overwrite='backup')