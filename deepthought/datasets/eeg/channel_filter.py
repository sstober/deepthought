__author__ = 'sstober'

class ChannelFilter(object):
    def keep_channel(self, channel_number):
        raise NotImplementedError('not implemented')

class NoChannelFilter(ChannelFilter):
    def keep_channel(self, channel_number):
        return True

class RemoveChannelsByNumber(ChannelFilter):
    def __init__(self, remove_channels):
        self.remove_channels = remove_channels

    def keep_channel(self, channel_number):
        if channel_number in self.remove_channels:
            return False
        else:
            return True

class KeepChannelsByNumber(ChannelFilter):
    def __init__(self, keep_channels):
        self.keep_channels = keep_channels

    def keep_channel(self, channel_number):
        if channel_number in self.keep_channels:
            return True
        else:
            return False
