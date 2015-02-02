__author__ = 'sstober'

import numpy as np

class ChannelLayout(object):

    def channel_names(self):
        raise NotImplementedError('not implemented')

    def num_channels(self):
        raise NotImplementedError('not implemented')

    def sphere_coords(self):
        raise NotImplementedError('not implemented')

    def sphere_radius(self):
        raise NotImplementedError('not implemented')

    def xyz_coords(self):
        raise NotImplementedError('not implemented')

    def get_channel_number(self, channel_name):
        raise NotImplementedError('not implemented')

    def projected_xy_coords(self):
        raise NotImplementedError('not implemented')

class XYPlotChannelLayout(object):

    def names_layout(self):
        raise NotImplementedError('not implemented')

    def numbers_layout(self):
        raise NotImplementedError('not implemented')

    def number_positions(self):
        raise NotImplementedError('not implemented')

