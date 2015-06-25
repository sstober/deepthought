__author__ = 'sstober'

import numpy as np

def compute_autocorrelation(data):
    # from pandas.tools.plotting import autocorrelation_plot
    # see http://pandas.pydata.org/pandas-docs/dev/visualization.html#autocorrelation-plot
    # see http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

    from pandas.compat import lmap

    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
    x = np.arange(n) + 1
    y = lmap(r, x)
    return np.asarray(y, dtype=np.float32)


def autocorr_index_to_bpm(index, sfreq):
    return 60.0*sfreq / index


def bpm_to_autocorr_index(bpm, sfreq):
    return 60.0*sfreq / bpm


def compute_tempo_histogram_from_ac(acorr, sfreq, mode='spline', plot=False):
    from scipy import interpolate

    x_samples = 1 + np.arange(0, len(acorr), 1) # use full length

    if mode == 'spline':
        spline_curve = interpolate.splrep(x_samples, acorr, s=0)

        def spline_tempo_curve(index):
            return interpolate.splev(index, spline_curve, der=0)

        interpolated = spline_tempo_curve

    elif mode == 'linear':
        interpolated = interpolate.interp1d(x_samples, acorr) # directly returns a function

    def tempo_curve(bpm):
        # convert bpm to ac index
        index = bpm_to_autocorr_index(bpm, sfreq)
#         print index
#         print bpm
#         print interpolated(index)
#         assert index >= x_samples[0] and index <= x_samples[-1]
        return interpolated(index)

    def enh_tempo_curve(bpm, octaves=[1], factors=None):
        if factors is None:
            factors = np.ones((len(octaves)))

        bpm = np.atleast_1d(bpm)
        result = np.zeros_like(bpm, dtype=float)
        for i in range(len(octaves)):
            octave = float(octaves[i])
            factor = float(factors[i])
            v = octave * bpm

            result += factor * tempo_curve(v)

        return result

    # plot tempo curve sample positions in original ac curve
    if plot:
        import matplotlib.pyplot as plt

    #     x_min = int(np.ceil(60 * sfreq / len(acorr)))
        bpm_min = int(np.ceil(autocorr_index_to_bpm(x_samples[-1], sfreq=sfreq)))
    #     x_max = int(np.floor(60 * sfreq))
        bpm_max = int(np.floor(autocorr_index_to_bpm(x_samples[0], sfreq=sfreq)))
        print bpm_min, bpm_max

        bpm_steps = np.arange(bpm_min, bpm_max, 1)
        x_bpm = bpm_to_autocorr_index(bpm_steps, sfreq)

        plt.figure(figsize=(10,3))
        plt.plot(x_samples, acorr)
        # for visual checking:
        for x in x_bpm:
            plt.axvline(x, color='gray', linestyle='--', linewidth=1) # zero

    return enh_tempo_curve