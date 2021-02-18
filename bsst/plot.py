# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Functions for plotting incidence, recruitment, and events in Metis."""

from metis import colors_config as cc
import matplotlib as mpl
import numpy as np
import pandas as pd
from metis import plot_utils
import warnings
from metis import ville_config

# All functions here take an axis argument and modify it in place.
# Functions in plot_utils return arguments.

pd.plotting.register_matplotlib_converters() # need to run on this pandas version

def turn_spines_off(ax, list_of_spines=['bottom', 'top', 'left', 'right']):
    """Turn off sides of the bounding box.

    Args:
        ax: The axis instance to format
        list_of_spines: A list of spines to turn off
    """
    for spine in list_of_spines:
        ax.spines[spine].set_visible(False)

def format_time_axis(ax, num_ticks=4, include_labels=True, date_format='%b'):
    """Formats the x axis to be time in datetime form, with num_ticks.

    Args:
        ax: The axis instance we want to format, we assume time is on the xaxis.
        num_ticks: An int representing the number of ticks to include on the final image.
        include_labels: Bool representing whether to include text labels
        date_format: A string representing how to format the date. '%b' is month
            only.
    """
    range = ax.get_xlim()
    step_size = (range[1] - range[0]) // (num_ticks - 1)
    ticks = range[0] + [i * step_size for i in np.arange(num_ticks)]

    dt_ticks = mpl.dates.num2date(ticks) if not np.issubdtype(ticks.dtype, np.datetime64) else ticks

    labels = [pd.to_datetime(x).strftime(date_format) for x in dt_ticks]

    ax.set_xticks(ticks=ticks)
    if include_labels:
        ax.set_xticklabels(labels=labels, rotation=30)
        ax.set_xlabel('date')
    else:
        ax.set_xticklabels(labels=[], visible=False)

def format_hist_time_axis(ax, bins, special_bins=[(0, '<'),  (-2, '>'),
                                                  (-1, 'Did not\nsucceed')],
                          num_ticks=4, include_labels=True, date_format='%b-%d'):
    """Formats the x axis to be time in datetime form, with num_ticks.

    Args:
        ax: The axis instance we want to format, we assume time is on the xaxis.
        bins: The bins used to plot the histogram
        special_bins: A series of tuples with the first entry representing the
            index of a bins with special values and the second entry
            representing the label to give to the bin.
        num_ticks: An int representing the number of ticks to include on the final image.
        include_labels: Bool representing whether to include text labels
        date_format: A string representing how to format the date. '%b' is month
            only.
    """
    eligible_bins = np.delete(bins, [idx for idx in [special_bins[i][0] for i in range(len(special_bins))]])
    # Don't put a tick at the far right, as we need to see the DnC label
    step_size = len(eligible_bins) // num_ticks
    ticks = eligible_bins[[i * step_size for i in np.arange(num_ticks)]]

    dt_ticks = mpl.dates.num2date(ticks) if not np.issubdtype(ticks.dtype, np.datetime64) else ticks
    labels = [pd.to_datetime(x).strftime(date_format) for x in dt_ticks]

    ticks = np.append(ticks, bins[[special_bins[i][0] for i in range(len(special_bins))]])

    labels = np.append(labels, [special_bins[i][1] for i in range(len(special_bins))])

    ax.set_xticks(ticks=ticks)
    if include_labels:
        ax.set_xticklabels(labels=labels, rotation=30)
        ax.set_xlabel('date')
    else:
        ax.set_xticklabels(labels=[], visible=False)

def array_over_time(ax, array_to_plot, first_plot_day=None, plot_kwargs={'color':'b', 'ls':'-'}):
    """Plot array_to_plot as a function of time.

    If array has a `sample` or `scenario` dimension, then all samples will be plotted with a
    low opacity (alpha) value.

    Args:
    ax: An axes instance to plot our data on.
    array_to_plot: A xr.DataArray with a time dimension, and optionally a sample OR
        scenario dimension
    first_plot_day: Optional, a time coordinate indicating the first date to plot.
    plot_kwargs: Optional, a dictionary with keyword arguments to pass to matplotlib.plot
    """
    time_dim = plot_utils.find_time_dim(array_to_plot)
    shaped_data = array_to_plot.transpose(time_dim, ...)

    if first_plot_day in array_to_plot.coords[time_dim].values:
        data = shaped_data.sel({time_dim:slice(first_plot_day, None)})
    else:
        data = shaped_data

    if any(item in data.dims for item in ['sample', 'scenario', 'sample_flattened']):
        alpha = 0.1
    else:
        alpha = 1.0

    ax.plot(data[time_dim], data.values, **plot_kwargs, alpha=alpha)

def cum_control_events(ax, control_events, first_plot_day, color, linestyle):
    """Plot cumulative control arm events over time or historical_time.

    Args:
        ax: The axis instance we want to plot on.
        control_events: The xr.DataArray that we want to plot. Must have either
            'time' OR 'historical_time' dimension.
        first_plot_day: An int representing the first date to plot.
        color: A mpl color.
        linestyle: A mpl linestyle.
    """
    time_dim = plot_utils.find_time_dim(control_events)
    cum_events = control_events.cumsum(time_dim)
    plot_array_over_time(ax, cum_events, first_plot_day, {'color': color, 'ls': linestyle})
    ax.set_ylabel('Cumulative control events')

def incidence(ax, incidence, first_plot_day, color, linestyle):
    """Plot incidence over time or historical_time.

    Args:
        ax: The axis instance we want to plot on.
        incidence: The xr.DataArray that we want to plot. Must have either
            'time' OR 'historical_time' dimension.
        first_plot_day: An int representing the first date to plot.
        color: A mpl color.
        linestyle: A mpl linestyle.
    """
    array_over_time(ax, incidence, first_plot_day, {'color': color, 'ls': linestyle})
    ax.set_ylabel('New cases / population')

def cum_recruits(ax, recruits, first_plot_day, color, linestyle):
    """Plot cumulative recruits over a time dimension.

    Args:
        ax: The axis instance we want to plot on.
        recruits: The xr.DataArray that we want to plot. Must have either
            'time' OR 'historical_time' dimension.
        first_plot_day: An int representing the first date to plot.
        color: A mpl color.
        linestyle: A mpl linestyle.
    """
    time_dim = plot_utils.find_time_dim(recruits)
    cum_recruits = recruits.cumsum(time_dim)
    array_over_time(ax, cum_recruits, first_plot_day, {'color': color, 'ls': linestyle})
    ax.set_ylabel('Cumulative recruits')

def cum_subrecruits(ax, recruits, first_plot_day, color, linestyle):
    """Plot the cumulative sum of recruits to compare across many populations.

    Args:
        ax: A series of axis instances to plot on.
        recruits: An xr.DataArray that representing the expected or
            observed recruits. Must have a time dimension.
        first_plot_day: An int representing the first date to plot.
        color: A mpl color.
        linestyle: A mpl linestyle.
    """
    sel_recruits = plot_utils.unpack_participant_labels(recruits)
    labels_to_plot = plot_utils.get_labels_to_plot(recruits)
    num_plots = len(ax)
    for i, label in enumerate(labels_to_plot):
        a = ax[i]
        participants = sel_recruits.sel(participant_label=label, drop=True)
        a.set_title(label)
        time_dim = plot_utils.find_time_dim(participants)
        array_over_time(a, participants.cumsum(time_dim), first_plot_day,
                        {'color': color, 'ls': linestyle})

        if i in [num_plots-2, num_plots-1]:
            format_time_axis(a, 3, date_format='%b-%d')
        else:
            format_time_axis(a, 3, include_labels=False)

def recruits(dim_to_plot, ax, sorted_recruits, color, linestyle='-', label=None):
    """Plot the recruits as a histogram over dim_to_plot.

    Args:
        dim_to_plot: A string representing the sorted_recruits.dim to plot
            along the x-axis.
        ax: An axes instance to plot our data on.
        sorted_recruits: A xr.DataArray representing the recruits to plot
            where <dim> has been sorted into the desired display order.
        color: A mpl color to use as the edgecolor
        linestyle: A mpl linestyle
        label: A string used as a plot label
    """
    dims_to_sum = list(sorted_recruits.dims)
    dims_to_sum.remove(dim_to_plot)
    thc = cc.TRANSPARENT_HIST_COLOR
    bh = cc.BAR_HEIGHT
    lw = cc.LINE_WIDTH

    sum_rec = sorted_recruits.sum(dims_to_sum)
    ax.barh(sum_rec[dim_to_plot], sum_rec, height=bh, fc=color, ec=thc,
            alpha=0.3, ls=linestyle, lw=lw, label=label)

def recruit_diffs(dim_to_plot, ax, sorted_recruits, recruits_left,
                  zero_left_edge=False):
    """Plot the difference between two sets of recruits.

    Places vertical lines at the actual recruitment value. Color maps and bar
    height read from colors_config.

    Args:
        dim_to_plot: A string representing the sorted_recruits.dim to plot
            along the x-axis.
        ax: An axes instance to plot our data on.
        sorted_recruits: A xr.DataArray representing the recruits to plot as the
            right edge of the bar chart.
            where <dim> has been sorted into the desired display order.
        recruits_left: A xr.DataArray representing the recruits to plot as the
            left edge of the bar chart.
        zero_left_edge: A boolean. If True, we plot the left edge at 0.
    """
    dims_to_sum = list(sorted_recruits.dims)
    dims_to_sum.remove(dim_to_plot)

    sum_rec_right = sorted_recruits.sum(dims_to_sum)
    sum_rec_left = recruits_left.sum(dims_to_sum)

    cmap = cc.BAR_CHART_CMAP
    norm = cc.BAR_CHART_NORM
    bh = cc.BAR_HEIGHT

    ax.set_facecolor(cc.BAR_CHART_FACECOLOR)

    # Sort the left edges to match the right edges
    ydim = sum_rec_right.dims[0]
    sorted_rec_left = sum_rec_left.sel({ydim: sorted_recruits[ydim]})

    diff = sum_rec_right - sorted_rec_left

    if not zero_left_edge:
        bar_plot = ax.barh(diff[ydim], diff, left=sorted_rec_left,
                           color=cmap(norm(diff.values)), height=bh)
        # Add vertical lines at the left-most edges
        lh = cc.VLINE_HEIGHT
        lc = cc.VLINE_COLOR
        # ycoord is lower left of box
        ycoords = np.asarray([i.xy[1] for i in bar_plot.get_children()])
        lower_lim = ycoords - .5 * (lh - bh)
        ax.vlines(sum_rec_right, lower_lim, lower_lim + lh, lc, ls='dashed')
    else:
        ax.barh(diff[ydim], diff, left=None, color=cmap(norm(diff.values)),
                height=bh)
        # Add a line at 0 to guide the eye
        ax.axvline(color='#000000', lw=1.0)

def tts(ax, events, efficacy, color, linestyle):
    """Plot the time to success distributions.

    Args:
        ax: The axis instance we want to plot on.
        events: An xr.DataArray representing the number of events in
            our control arm. Has dimensions (time, location, scenario)
        efficacy: A float representing the assumed vaccine efficacy.
        color: A mpl color for the bar faces
        linestyle: A mpl linestyle for the bar edges
    """
    lw = cc.LINE_WIDTH
    thc = cc.TRANSPARENT_HIST_COLOR
    ax.set_facecolor(thc)
    hist, bins = plot_utils.make_tts_hist(events, efficacy)
    bw = bins[1] - bins[0]
    ax.bar(bins[:-1], hist, width=bw, align='edge',
           fc=color, ec=thc, ls=linestyle, lw=lw, alpha=0.3)
    format_hist_time_axis(ax, bins[:-1], date_format='%b-%d')
    ax.axvline(x=bins[-2], color='#656565', lw=1.0, ls='--')


def tts_diff(ax, proposed_events, baseline_events, efficacy):
    """Plot the difference in time to success distributions.

    Args:
        ax: The axis instance we want to plot on.
        proposed_events: An xr.DataArray representing the number of events in
            our control arm. Has dimensions (time, location, scenario)
        baseline_events: An xr.DataArray representing the baseline number of
            control events. This becomes the bottom edge of the barh plot.
            Has dimensions (time, location, scenario)
        efficacy: A float representing the assumed vaccine efficacy.
    """
    ax.set_facecolor(cc.BAR_CHART_FACECOLOR)

    cmap = cc.BAR_CHART_CMAP
    norm = cc.DIFF_NORM

    proposed_hist, proposed_bins = plot_utils.make_tts_hist(proposed_events, efficacy)
    baseline_hist, baseline_bins = plot_utils.make_tts_hist(baseline_events, efficacy)

    if np.any(proposed_bins[~np.isnan(proposed_bins)] != baseline_bins[~np.isnan(baseline_bins)]):
        warnings.warn(f'Proposed and baseline events have different times.')

    diff = proposed_hist - baseline_hist
    bw = proposed_bins[1] - proposed_bins[0]

    ax.bar(proposed_bins[:-1], diff, width=bw, align='edge',
            color=cmap(norm(diff)))

    # Add a line at 0 to guide the eye
    ax.axhline(color='#000000', lw=1.0)

    # add line on the dns bin
    ax.axvline(x=proposed_bins[-2], color='#656565', lw=1.0, ls='--')

    # Format time axis here
    format_hist_time_axis(ax, proposed_bins[:-1], date_format='%b-%d')
