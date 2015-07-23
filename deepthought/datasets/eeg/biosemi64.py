
__author__ = 'sstober'

import numpy as np

from deepthought.datasets.eeg.channel_layout import ChannelLayout, XYPlotChannelLayout

class ChannelNameLoader(object):
    """
    loader for use with !obj in YAML files
    """
    def __new__(cls, *args, **kwargs):
        return Biosemi64Layout().channel_names()


class Biosemi64Layout(ChannelLayout):
    """
    Information about the 64 channel biosemi setup

    All information has been taken from
    http://www.biosemi.com/download/Cap_coords_all.xls
    """
    def channel_names(self):
        # EEG_CHANNEL_NAMES = \
        return np.asarray([
                'Fp1','AF7','AF3','F1','F3','F5','F7','FT7',
                'FC5','FC3','FC1','C1','C3','C5','T7','TP7',
                'CP5','CP3','CP1','P1','P3','P5','P7','P9',
                'PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
                'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4',
                'F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
                'C2','C4','C6','T8','TP8','CP6','CP4','CP2',
                'P2','P4','P6','P8','P10','PO8','PO4','O2'
            ])

    def num_channels(self):
        return 64

    def sphere_coords(self):
        """
        Spherical coordinates (Inclination (theta), Azimuth (phi))
        The spherical coordinates are in degrees, the head is approximated as a sphere.
        Spherical angles are measured from an imaginary center in the middle of the head.
        Inclination:
                positive is right hemisphere,
                directing Cz is 0,
                negative is left hemisphere.
        Azimuth:
                from T7 for left hemisphere, and
                from T8 for the right hemisphere,
                pos is anti-clockwise, neg is clockwise

        http://en.wikipedia.org/wiki/Spherical_coordinate_system
        """
        # EEG_CHANNEL_SPHERE_COORDS = [
        return np.asarray([
                [-92, -72],[-92, -54],[-74, -65],[-50, -68],
                [-60, -51],[-75, -41],[-92, -36],[-92, -18],
                [-72, -21],[-50, -28],[-32, -45],[-23, 0],
                [-46, 0],[-69, 0],[-92, 0],[-92, 18],
                [-72, 21],[-50, 28],[-32, 45],[-50, 68],
                [-60, 51],[-75, 41],[-92, 36],[-115, 36],
                [-92, 54],[-74, 65],[-92, 72],[115, -90],
                [92, -90],[69, -90],[46, -90],[23, -90],
                [92, 90],[92, 72],[92, 54],[74, 65],
                [69, 90],[46, 90],[50, 68],[60, 51],
                [75, 41],[92, 36],[92, 18],[72, 21],
                [50, 28],[32, 45],[23, 90],[0, 0],
                [23, 0],[46, 0],[69, 0],[92, 0],
                [92, -18],[72, -21],[50, -28],[32, -45],
                [50, -68],[60, -51],[75, -41],[92, -36],
                [115, -36],[92, -54],[74, -65],[92, -72]
            ], dtype=np.int)


    def sphere_radius(self):
        100.0

    def xyz_coords(self):
        """
        Cartesian coordinates (x, y, z)
        Below Cartesian values are in length, with r chosen at 100.
        The center of the head is located at position 0,0,0.
        x: positive is direction neck.
        y: positive is direction right ear.
        z: positive is direction sky.

        http://en.wikipedia.org/wiki/Cartesian_coordinate_system
        """
        # EEG_CHANNEL_XYZ_COORDS = [
        return np.asarray([
                [-95.04771584, -30.88287496, -3.48994967],
                [-80.85241631, -58.74271894, -3.48994967],
                [-87.11989604, -40.6246747, 27.56373558],
                [-71.02640395, -28.69652992, 64.27876097],
                [-67.30281451, -54.50074458, 50],
                [-63.37043597, -72.89934749, 25.88190451],
                [-58.74271894, -80.85241631, -3.48994967],
                [-30.88287496, -95.04771584, -3.48994967],
                [-34.08281736, -88.78877481, 30.90169944],
                [-35.96360819, -67.63770971, 64.27876097],
                [-37.47095052, -37.47095052, 84.80480962],
                [0, -39.07311285, 92.05048535],
                [0, -71.93398003, 69.46583705],
                [0, -93.35804265, 35.83679495],
                [0, -99.9390827, -3.48994967],
                [30.88287496, -95.04771584, -3.48994967],
                [34.08281736, -88.78877481, 30.90169944],
                [35.96360819, -67.63770971, 64.27876097],
                [37.47095052, -37.47095052, 84.80480962],
                [71.02640395, -28.69652992, 64.27876097],
                [67.30281451, -54.50074458, 50],
                [63.37043597, -72.89934749, 25.88190451],
                [58.74271894, -80.85241631, -3.48994967],
                [53.27143513, -73.32184018, -42.26182617],
                [80.85241631, -58.74271894, -3.48994967],
                [87.11989604, -40.6246747, 27.56373558],
                [95.04771584, -30.88287496, -3.48994967],
                [90.6307787, 5.55181E-15, -42.26182617],
                [99.9390827, 6.12201E-15, -3.48994967],
                [93.35804265, 5.71887E-15, 35.83679495],
                [71.93398003, 4.40649E-15, 69.46583705],
                [39.07311285, 2.39352E-15, 92.05048535],
                [-99.9390827, 6.12201E-15, -3.48994967],
                [-95.04771584, 30.88287496, -3.48994967],
                [-80.85241631, 58.74271894, -3.48994967],
                [-87.11989604, 40.6246747, 27.56373558],
                [-93.35804265, 5.71887E-15, 35.83679495],
                [-71.93398003, 4.40649E-15, 69.46583705],
                [-71.02640395, 28.69652992, 64.27876097],
                [-67.30281451, 54.50074458, 50],
                [-63.37043597, 72.89934749, 25.88190451],
                [-58.74271894, 80.85241631, -3.48994967],
                [-30.88287496, 95.04771584, -3.48994967],
                [-34.08281736, 88.78877481, 30.90169944],
                [-35.96360819, 67.63770971, 64.27876097],
                [-37.47095052, 37.47095052, 84.80480962],
                [-39.07311285, 2.39352E-15, 92.05048535],
                [0, 0, 100],
                [0, 39.07311285, 92.05048535],
                [0, 71.93398003, 69.46583705],
                [0, 93.35804265, 35.83679495],
                [0, 99.9390827, -3.48994967],
                [30.88287496, 95.04771584, -3.48994967],
                [34.08281736, 88.78877481, 30.90169944],
                [35.96360819, 67.63770971, 64.27876097],
                [37.47095052, 37.47095052, 84.80480962],
                [71.02640395, 28.69652992, 64.27876097],
                [67.30281451, 54.50074458, 50],
                [63.37043597, 72.89934749, 25.88190451],
                [58.74271894, 80.85241631, -3.48994967],
                [53.27143513, 73.32184018, -42.26182617],
                [80.85241631, 58.74271894, -3.48994967],
                [87.11989604, 40.6246747, 27.56373558],
                [95.04771584, 30.88287496, -3.48994967]
            ])


    __EEG_CHANNEL_NAME_TO_NUMBER = {
        'POz': 29, 'FC1': 10, 'Pz': 30, 'FCz': 46, 'C4': 49, 'FC3': 9, 'P6': 58, 'O2': 63,
        'O1': 26, 'P7': 22, 'P4': 57, 'P10': 60, 'T8': 51, 'FT7': 7, 'FT8': 42, 'Fz': 37,
        'TP7': 15, 'CPz': 31, 'AF8': 34, 'PO7': 24, 'C3': 12, 'C2': 48, 'C1': 11, 'AF7': 1,
        'C6': 50, 'C5': 13, 'AF3': 2, 'P2': 56, 'P3': 20, 'FC2': 45, 'P1': 19, 'FC4': 44,
        'FC5': 8, 'FC6': 43, 'P5': 21, 'T7': 14, 'P8': 59, 'P9': 23, 'PO3': 25, 'PO4': 62,
        'AF4': 35, 'Iz': 27, 'Fp1': 0, 'Oz': 28, 'Fp2': 33, 'TP8': 52, 'F4': 39, 'PO8': 61,
        'F1': 3, 'F2': 38, 'F3': 4, 'Fpz': 32, 'F5': 5, 'F6': 40, 'F7': 6, 'F8': 41,
        'Cz': 47, 'AFz': 36, 'CP1': 18, 'CP2': 55, 'CP3': 17, 'CP4': 54, 'CP5': 16, 'CP6': 53
    }

    def get_channel_number(self, channel_name):
        return self.__EEG_CHANNEL_NAME_TO_NUMBER[channel_name]


    def get_default_xy_plot_layout(self):
        return Biosemi64XYPlotChannelLayout()

    __PROJECTED_XY_COORDS = None
    def projected_xy_coords(self):
        # lazy init
        if self.__PROJECTED_XY_COORDS is None:
            sphere_coords = self.sphere_coords()
            pos = np.empty_like(sphere_coords, dtype=np.float64)

            dist = sphere_coords[:,0] / 90.0
            angle = sphere_coords[:,1]

            # compensate stupid sign use -> neg values are left hemisphere
            for i in xrange(64):
                if dist[i] < 0:
                    dist[i] = -dist[i]
                    angle[i] += 180.0

                if angle[i] < 0:
                    angle[i] += 360.0

            # print 'polar coordinates:'
            # for i in xrange(64):
            #     print '{}: {:.2f} {:.2f}'.format(self.channel_names()[i], angle[i], dist[i])

            # convert to radians
            rad = np.pi * angle / 180.0

            pos[:,0] = dist * np.cos(rad)
            pos[:,1] = dist * np.sin(rad)

            # print 'XY coordinates:'
            # for i in xrange(64):
            #     print '{}: {:.2f} {:.2f}'.format(channel_names[i], pos[i,0], pos[i,1])
            #     # print '[{:.4f}, {:.4f}],'.format(pos[i,0], pos[i,1])
            self.__PROJECTED_XY_COORDS = pos

        return self.__PROJECTED_XY_COORDS

    def as_montage(self):
        from mne.channels.montage import Montage
        pos = self.xyz_coords()
        # print pos[1,:]
        pos = pos[:,[1,0,2]]  # swap x and y
        # print pos[1,:]
        pos *= [1, -1, 1]  # flip y
        # print pos[1,:]
        montage = Montage(pos=pos,
                          ch_names=self.channel_names(),
                          kind='64channels',                # FIXME: no idea what this should be
                          selection=list(range(64)))
        return montage

# EXT_CHANNEL_NAMES = ['EXT1', 'EXT2', 'EXT3', 'EXT4']
#
# NUM_EEG_CHANNELS = len(EEG_CHANNEL_NAMES)
# NUM_EXT_CHANNELS = len(EXT_CHANNEL_NAMES)
# NUM_CHANNELS = NUM_EEG_CHANNELS + NUM_EXT_CHANNELS


class Biosemi64XYPlotChannelLayout(XYPlotChannelLayout):
    """
    precomputed 2-d layout for plotting
    """

    __NAMES_LAYOUT = np.asarray([
            ['  ', '   ', '   ', '   ', 'Fp1', 'Fpz', 'Fp2', '   ', '   ', '   ', '   '],
            ['  ', '   ', 'AF7', 'AF3', '   ', 'AFz', '   ', 'AF4', 'AF8', '   ', '   '],
            ['  ', 'F7' , 'F5' , 'F3' , 'F1' , 'Fz' , 'F2' , 'F4' , 'F6' , 'F8' , '   '],
            ['  ', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', '   '],
            ['  ', 'T7' , 'C5' , 'C3' , 'C1' , 'Cz' , 'C2' , 'C4' , 'C6' , 'T8' , '   '],
            ['  ', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', '   '],
            ['P9', 'P7' , 'P5' , 'P3' , 'P1' , 'Pz' , 'P2' , 'P4' , 'P6' , 'P8' , 'P10'],
            ['  ', '   ', 'PO7', 'PO3', '   ', 'POz', '   ', 'PO4', 'PO8', '   ', '   '],
            ['  ', '   ', '   ', '   ', 'O1' , 'Oz' , 'O2' , '   ', '   ', '   ', '   '],
            ['  ', '   ', '   ', '   ', '   ', 'Iz' , '   ', '   ', '   ', '   ', '   ']
        ]).swapaxes(0,1) # x should be 1st dimension

    __NUMBERS_LAYOUT = np.asarray([
            [-1, -1, -1, -1,  0, 32, 33, -1, -1, -1, -1],
            [-1, -1,  1,  2, -1, 36, -1, 35, 34, -1, -1],
            [-1,  6,  5,  4,  3, 37, 38, 39, 40, 41, -1],
            [-1,  7,  8,  9, 10, 46, 45, 44, 43, 42, -1],
            [-1, 14, 13, 12, 11, 47, 48, 49, 50, 51, -1],
            [-1, 15, 16, 17, 18, 31, 55, 54, 53, 52, -1],
            [23, 22, 21, 20, 19, 30, 56, 57, 58, 59, 60],
            [-1, -1, 24, 25, -1, 29, -1, 62, 61, -1, -1],
            [-1, -1, -1, -1, 26, 28, 63, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 27, -1, -1, -1, -1, -1]
        ], dtype=int).swapaxes(0,1) # x should be 1st dimension

    __NUMBER_POSITIONS = np.asarray([
        [ 4, 0],[ 2, 1],[ 3, 1],[ 4, 2],[ 3, 2],[ 2, 2],[ 1, 2],[ 1, 3],
        [ 2, 3],[ 3, 3],[ 4, 3],[ 4, 4],[ 3, 4],[ 2, 4],[ 1, 4],[ 1, 5],
        [ 2, 5],[ 3, 5],[ 4, 5],[ 4, 6],[ 3, 6],[ 2, 6],[ 1, 6],[ 0, 6],
        [ 2, 7],[ 3, 7],[ 4, 8],[ 5, 9],[ 5, 8],[ 5, 7],[ 5, 6],[ 5, 5],
        [ 5, 0],[ 6, 0],[ 8, 1],[ 7, 1],[ 5, 1],[ 5, 2],[ 6, 2],[ 7, 2],
        [ 8, 2],[ 9, 2],[ 9, 3],[ 8, 3],[ 7, 3],[ 6, 3],[ 5, 3],[ 5, 4],
        [ 6, 4],[ 7, 4],[ 8, 4],[ 9, 4],[ 9, 5],[ 8, 5],[ 7, 5],[ 6, 5],
        [ 6, 6],[ 7, 6],[ 8, 6],[ 9, 6],[10, 6],[ 8, 7],[ 7, 7],[ 6, 8],
    ], dtype=int)

    def names_layout(self):
        return self.__NAMES_LAYOUT.copy()

    def numbers_layout(self):
        return self.__NUMBERS_LAYOUT.copy()

    def number_positions(self):
        return self.__NUMBER_POSITIONS.copy()

    def apply_to(self, data):
        output = np.zeros_like(self.__NUMBERS_LAYOUT, dtype=data.dtype)
        for i, pos in enumerate(self.__NUMBER_POSITIONS):
            output[pos[0], pos[1]] = data[i]
        return output




