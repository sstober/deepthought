__author__ = 'sstober'

import numpy as np

# default settings
from deepthought.datasets.eeg.biosemi64 import Biosemi64Layout as Biosemi64


def compute_plot_xy_mapping(xy_layout, layout=Biosemi64()):

    channel_xy_positions = np.zeros((layout.num_channels(),2), dtype=np.int)
    channel_name_to_number = dict()
    for number, name in enumerate(layout.channel_names()):
        channel_name_to_number[name] = number

    print xy_layout.shape

    channel_xy_numbers = np.full(layout.shape, -1, dtype=np.int)
    for x in xrange(layout.shape[0]):
        for y in xrange(layout.shape[1]):
            name = layout[x,y].strip()
            if name != '':
                number = channel_name_to_number[name]
                channel_xy_positions[number,:] = [x,y]
                channel_xy_numbers[x,y] = number

    return channel_name_to_number, channel_xy_positions, channel_xy_numbers


def electrode_distance(latlong_a, latlong_b, radius=100.0):
    """
    geodesic (great-circle) distance between two electrodes, A and B
    :param latlong_a: spherical coordinates of electrode A
    :param latlong_b: spherical coordinates of electrode B
    :return: distance
    """
    import math

    lat1, lon1 = latlong_a
    lat2, lon2 = latlong_b

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dLon / 2) *
            math.sin(dLon / 2))

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def compute_electrode_distance_matrix(layout):
    sphere_coords = layout.sphere_coords()
    radius = layout.sphere_radius()

    import numpy as np
    num_channels = len(sphere_coords)
    d = np.zeros((num_channels, num_channels))
    for i in xrange(len(sphere_coords)):
        for j in xrange(i+1, len(sphere_coords)):
            d[i,j] = electrode_distance(
                            sphere_coords[i],
                            sphere_coords[j],
                            radius
                        )
            d[j,i] = d[i,j]
    return d

def compute_2d_mapping(layout):
    sphere_coords = layout.sphere_coords()
    radius = layout.sphere_radius()
    from sklearn.manifold import MDS
    distances = compute_electrode_distance_matrix(sphere_coords, radius)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    projection = mds.fit_transform(distances)
    # print projection.shape
    return projection

def reorder_channels_by_xyz_coord(data, channel_names=None):
    """
    :param data: 2-d array in the format [n_samples, n_channels]
    :param channel_names: names of the EEG channels
    :return: data, channel_names permutated accordingly
    """
    # work on transposed view, i.e. [channel, samples]
    data = data.T

    # map channels to 1-d coordinates through MDS
    from sklearn.manifold import MDS
    distances = compute_electrode_distance_matrix()
    mds = MDS(n_components=1, dissimilarity='precomputed')
    projection = mds.fit_transform(distances).reshape(data.shape[0])
    order = np.argsort(projection)
    print mds.stress_
    print order

    # re-order channels
    data = data[order]
    # restore initial axes layout
    data = data.T

    # re-order channel_names
    channel_names = reorder_channel_names(channel_names, order)

    return data, channel_names

def reorder_channel_names(channel_names, order):
    if channel_names is not None:
        tmp = []
        for i in xrange(len(order)):
            tmp.append(channel_names[order[i]])
        channel_names = tmp
        # print channel_names
    return channel_names


def reorder_channels_by_similarity(data, channel_names=None, normalize=True):
    """
    :param data: 2-d array in the format [n_samples, n_channels]
    :param channel_names: names of the EEG channels
    :param normalize: if True, normalize data first before computing distances
    :return: data, channel_names permutated accordingly
    """

    # work on transposed view
    data = data.T

    # normalize first
    if normalize:
        # work on a copy
        data_copy = data.copy()
        for c in xrange(data_copy.shape[0]):
            data_copy[c] -= data_copy[c].mean()
            data_copy[c] /= data_copy[c].std()
    else:
        data_copy = data

    # project to 1-d
    from sklearn.manifold import MDS
    mds = MDS(n_components=1)
    projection = mds.fit_transform(data_copy).reshape(data_copy.shape[0])
    # print p.shape

    order = np.argsort(projection)
    # print order

    # the operation is not happening "in-place": a copy of the
    # subarray in sorted order is made, and then its contents
    # are written back to the original array
    data = data[order]

    # restore initial axes layout
    data = data.T

    # re-order channel_names
    channel_names = reorder_channel_names(channel_names, order)

    return data, channel_names