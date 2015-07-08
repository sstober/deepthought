__author__ = 'sstober'

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from mne.io.pick import channel_type
from mne.externals.six import string_types
from mne.defaults import _handle_default
from mne.viz.utils import tight_layout

def plot_ica_overlay_evoked(evoked, evoked_cln, title, show):
    """
    workaround for https://github.com/mne-tools/mne-python/issues/1819
    copied from mne.viz.ica._plot_ica_overlay_evoked()

    Plot evoked after and before ICA cleaning

    Parameters
    ----------
    ica : instance of mne.preprocessing.ICA
        The ICA object.
    epochs : instance of mne.Epochs
        The Epochs to be regarded.
    show : bool
        If True, all open plots will be shown.

    Returns
    -------
    fig : instance of pyplot.Figure
    """
    ch_types_used = [c for c in ['mag', 'grad', 'eeg'] if c in evoked]
    n_rows = len(ch_types_used)
    ch_types_used_cln = [c for c in ['mag', 'grad', 'eeg'] if
                         c in evoked_cln]

    if len(ch_types_used) != len(ch_types_used_cln):
        raise ValueError('Raw and clean evokeds must match. '
                         'Found different channels.')

    fig, axes = plt.subplots(n_rows, 1)
    fig.suptitle('Average signal before (red) and after (black) ICA')
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

    evoked.plot(axes=axes, show=False)

    for ax in fig.axes:
        [l.set_color('r') for l in ax.get_lines()]

    fig.canvas.draw()
    evoked_cln.plot(axes=axes, show=show)
    tight_layout(fig=fig)

    if show:
        plt.show()

    fig.subplots_adjust(top=0.90)
    fig.canvas.draw()


def _plot_evoked(evoked, plot_type, colorbar=True, hline=None, ylim=None,
                picks=None, exclude='bads', unit=True, show=True,
                      clim=None, proj=False, xlim='tight', units=None,
                      scalings=None, titles=None, axes=None, cmap='RdBu_r'):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings)

    Extra param is:

    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    """
    import matplotlib.pyplot as plt
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')

    scalings = _handle_default('scalings', scalings)
    titles = _handle_default('titles', titles)
    units = _handle_default('units', units)

    channel_types = set(key for d in [scalings, titles, units] for key in d)
    channel_types = sorted(channel_types)  # to guarantee consistent order

    if picks is None:
        picks = list(range(evoked.info['nchan']))

    bad_ch_idx = [evoked.ch_names.index(ch) for ch in evoked.info['bads']
                  if ch in evoked.ch_names]
    if len(exclude) > 0:
        if isinstance(exclude, string_types) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list)
              and all([isinstance(ch, string_types) for ch in exclude])):
            exclude = [evoked.ch_names.index(ch) for ch in exclude]
        else:
            raise ValueError('exclude has to be a list of channel names or '
                             '"bads"')

        picks = list(set(picks).difference(exclude))

    types = [channel_type(evoked.info, idx) for idx in picks]
    n_channel_types = 0
    ch_types_used = []
    for t in channel_types:
        if t in types:
            n_channel_types += 1
            ch_types_used.append(t)

    axes_init = axes  # remember if axes where given as input

    fig = None
    if axes is None:
        fig, axes = plt.subplots(n_channel_types, 1)

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if axes_init is not None:
        fig = axes[0].get_figure()

    if not len(axes) == n_channel_types:
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%g)' % (len(axes), n_channel_types))

    # instead of projecting during each iteration let's use the mixin here.
    if proj is True and evoked.proj is not True:
        evoked = evoked.copy()
        evoked.apply_proj()

    times = 1e3 * evoked.times  # time in miliseconds
    for ax, t in zip(axes, ch_types_used):
        ch_unit = units[t]
        this_scaling = scalings[t]
        if unit is False:
            this_scaling = 1.0
            ch_unit = 'NA'  # no unit
        idx = [picks[i] for i in range(len(picks)) if types[i] == t]
        if len(idx) > 0:
            # Parameters for butterfly interactive plots
            if plot_type == 'butterfly':
                if any([i in bad_ch_idx for i in idx]):
                    colors = ['k'] * len(idx)
                    for i in bad_ch_idx:
                        if i in idx:
                            colors[idx.index(i)] = 'r'

                    ax._get_lines.color_cycle = iter(colors)
                else:
                    ax._get_lines.color_cycle = cycle(['k'])
            # Set amplitude scaling
            D = this_scaling * evoked.data[idx, :]
            # plt.axes(ax)
            if plot_type == 'butterfly':
                ax.plot(times, D.T)
            elif plot_type == 'image':
                im = ax.imshow(D, interpolation='nearest', origin='lower',
                               extent=[times[0], times[-1], 0, D.shape[0]],
                               aspect='auto', cmap=cmap)
                if colorbar:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.ax.set_title(ch_unit)
            elif plot_type == 'mean' :
#                 ax.plot(times, D.mean(axis=0))
                ax.plot(times, np.abs(D).mean(axis=0))
            if xlim is not None:
                if xlim == 'tight':
                    xlim = (times[0], times[-1])
                ax.set_xlim(xlim)
            if ylim is not None and t in ylim:
                if plot_type == 'butterfly' or plot_type == 'mean':
                    ax.set_ylim(ylim[t])
                elif plot_type == 'image':
                    im.set_clim(ylim[t])
            ax.set_title(titles[t] + ' (%d channel%s)' % (
                         len(D), 's' if len(D) > 1 else ''))
            ax.set_xlabel('time (ms)')
            if plot_type == 'butterfly' or plot_type == 'mean':
                ax.set_ylabel('data (%s)' % ch_unit)
            elif plot_type == 'image':
                ax.set_ylabel('channels (%s)' % 'index')
            else:
                raise ValueError("plot_type has to be 'butterfly' or 'image'."
                                 "Got %s." % plot_type)

            if (plot_type == 'butterfly' or plot_type == 'mean') and (hline is not None):
                for h in hline:
                    ax.axhline(h, color='r', linestyle='--', linewidth=2)

    if axes_init is None:
        plt.subplots_adjust(0.175, 0.08, 0.94, 0.94, 0.2, 0.63)

    # if proj == 'interactive':
    #     _check_delayed_ssp(evoked)
    #     params = dict(evoked=evoked, fig=fig, projs=evoked.info['projs'],
    #                   axes=axes, types=types, units=units, scalings=scalings,
    #                   unit=unit, ch_types_used=ch_types_used, picks=picks,
    #                   plot_update_proj_callback=_plot_update_evoked,
    #                   plot_type=plot_type)
    #     _draw_proj_checkbox(None, params)

    if show and plt.get_backend() != 'agg':
        plt.show()
        fig.canvas.draw()  # for axes plots update axes.
    tight_layout(fig=fig)

    return fig