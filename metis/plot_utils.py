# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Utils for plotting arrays in Metis."""

from metis import colors_config as cc
from metis import evaluation
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from metis import ville_config
import warnings
import xarray as xr

# All functions return something, those in plot.py only modify in place.

def new_figure(plt_size=(5, 5)):
    """Set up a single figure/axis of size plt_size.

    Note that we set it up using matplotlib.Figure instead of
    plt.figure. This avoids opening the figure until it is
    explicitly displayed.

    Args:
        plt_size: a tuple of floats representing the figure size.
    Returns:
        fig: A figure.Figure.
        axis: A axes to plot on.
    """
    fig = Figure(figsize=plt_size)
    axis = fig.subplots()
    canvas = FigureCanvasAgg(fig)
    return fig, axis

def make_subplots(num_rows, num_cols, plt_size=(5, 5),
                  sharex=True, sharey=True):
    """Make a figure with subplots.

    We do not use plt.subplots as that draws the figure everytime. This method
    creates the subplots without redrawing. It only redraws when display is
    called.

    Args:
        num_rows: An int representing the number of rows to add.
        num_cols: An int representing the number of columns to add.
        plt_size: A tuple of floats representing the figure size.
        sharex: A boolean indicating whether all subplots share the x-axis scale
        sharey: A boolean indicating whether all subplots share the y-axis scale

    Returns:
        fig: The figure instance with the subplots
        a: A list of axes corresponding to each subplot
    """
    fig = Figure(figsize=plt_size)
    axis = fig.subplots(num_rows, num_cols, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(wspace=0.75, hspace=0.75)
    canvas = FigureCanvasAgg(fig)

    a = axis.flatten()
    for i in range(len(a)):
        a[i].spines['right'].set_visible(False)
        a[i].spines['top'].set_visible(False)
    return fig, a

def find_time_dim(da):
    """Find the time dimension in an xr.DataArray.

    Args:
        da: An xr.DataArray
    Returns:
        time_dim: Either 'time' or 'historical_time', depending on which is
            present in da.
    """
    if 'time' in da.coords:
        time_dim = 'time'
    elif 'historical_time' in da.coords:
        time_dim = 'historical_time'
    else:
        time_dim = None
    return time_dim

def make_tts_hist(control_arm_events, efficacy):
    """Calculate histogram values of times to trial success.

    Assume a vaccine efficacy of efficacy. We calculate the number of needed
    events. Given the believed future summarized in control_arm_events,
    calculate the distribution of days on which we observe the needed events.
    Plot the histogram with success day in range ville_config.SUCCESS_DATES.
    Inlcude under/overflow and NaN bins to indicate scenarios that do not finish
    within the given success day range.

    Args:
        control_arm_events: An xr.DataArray representing the number of events
            in our control arm. Has dimensions (time, location, scenario)
        efficacy: A float representing the assumed vaccine efficacy.
    Returns:
        hist, bins: The histogram counts and the bins for the calculated
            times to success
    """
    if 'location' in control_arm_events.dims:
        # We'll never look at time to success in a single location...
        control_arm_events = control_arm_events.sum('location')

    needed_events = evaluation.needed_control_arm_events(efficacy)
    success_day = evaluation.success_day(needed_events, control_arm_events)

    # convert time from np.datetime to int to make numpy happy
    int_success_day = mpl.dates.date2num(success_day)

    # Zoom in on smaller time range
    tts_time = ville_config.SUCCESS_DATES
    time_trim = control_arm_events.sel(time=tts_time).time.values

    bw = ville_config.HIST_BIN_WIDTH
    raw_bins = mpl.dates.date2num(time_trim[::bw])

    # Add under/overflow bins
    bins = np.insert(raw_bins, 0, raw_bins[0]-bw)
    bins = np.append(bins, raw_bins[-1]+bw)

    # Add NaN to capture scenarios that didn't finish
    bins = np.append(bins, np.nan)

    # Finally, clip histogram to be in this range
    clip_success_day = np.clip(int_success_day, bins[0], bins[-2])
    return np.histogram(clip_success_day, bins)

def unpack_participant_labels(da):
    """Unpack participant labels in an xr.DataArray.

    Participant labels are stored in a grid, where the sum across each dim is 1.
    To get all participants of a certain age, we must sum over the other dims.
    This function sums across other dims, and returns a flattened array,
    where each rec_dims.coord is a participant_label.coord.

    Args:
        da: An xr.DataArray. We assume that any dims that are not in ['time',
            'historical_time', 'location'] are participant labels

    Returns:
        unpacked_da: A copy of xr.DataArray with the participant dims flattened
            into a single 'participant_label' dimension.
    """
    p_dims = list(da.dims)
    for dim in ['location', find_time_dim(da)]:
        if dim in p_dims: p_dims.remove(dim)

    p_list = []
    for dim in p_dims:
        sum_dims = p_dims.copy()
        sum_dims.remove(dim)
        sum_da = da.sum(sum_dims).rename({dim: 'participant_label'})
        p_list.append(sum_da)

    unpacked_da = xr.merge(p_list).to_array('p').squeeze('p')
    return unpacked_da

def get_labels_to_plot(da):
    """Returns a list of participant labels to plot.

    Args:
        da: An xr.DataArray. We assume that any dims that are not in ['time',
            'historical_time', 'location'] are participant labels
    Returns:
        labels_to_plot: A list of participant labels to plot. Removes any labels
            that are in ville_config.LABELS_TO_DROP.
    """
    unpack_da = unpack_participant_labels(da)
    labels_to_plot = list(unpack_da.participant_label.values)
    for l in ville_config.LABELS_TO_DROP:
        labels_to_plot.remove(l)

    return labels_to_plot
