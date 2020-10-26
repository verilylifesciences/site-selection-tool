# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Vis to compare the impact of site activations on trial outcomes."""

import colors_config
import datetime
import interactive_utils as int_utils
import ipywidgets as widgets
import functools
import metis.io
import numpy as np
import os
import optimization
import pathlib
import plot
import plot_utils
import sim
import sim_scenarios
import table_utils
import ville_config

"""An interactive exploration of how site activations impact trial outcomes.

This module loads in a ville in the form of an xr.Dataset. It then processes
the dataset to add the initial participants, site_activations, and events as
`original` quantities. The user can then change the site_activations for
individual sites, and we display the changes from original in
participant demographics and time to success (as determined by assumptions
in the input files).
"""

def site_activation(original_ville):
    """Make the site activation display.

    Args:
        original_ville: An xr.Dataset containing the ville as it is now.
            This is what all changes will be compared against.

    Returns:
        widget: An ipywidget with the display.
    """

    # TODO we are using the same scenarios throughout this ville
    # Add a button to regenerate the scenarios & rerun
    set_up_dataset(original_ville)

    str_time_start = np.datetime_as_string(original_ville.time.values[0], unit='D')
    str_time_end = np.datetime_as_string(original_ville.time.values[-1], unit='D')
    print(f'Only showing recruitment and events for {str_time_start} to '
          f'{str_time_end}')
    print(sim.recruitment_rules(original_ville))

    # Set up a status button to display when code is running
    # All update/redraw calls will need this status_button
    status_button = int_utils.new_status_button()

    table_box = widgets.VBox(children=[widgets.Output(), widgets.Output()])
    table_dropdown = make_table_button(original_ville, table_box, status_button)
    disp_table(original_ville, table_box, table_dropdown.value)

    # reorganize a bit
    org_table_box = widgets.VBox(children=[widgets.HBox(children=[table_box.children[0], table_dropdown]),
                                 table_box.children[1]])

    summary_box = widgets.HBox(children=[widgets.Output(), widgets.Output()])
    summary_plots(original_ville, summary_box)
    summary_box.layout.border = 'solid 1px gray'

    loc_box = widgets.HBox(children=[widgets.Output(), widgets.Output(), widgets.Output()])
    loc_plots(original_ville, loc_box)

    button_box = make_loc_buttons(original_ville, summary_box, loc_box,
                                  table_box, table_dropdown, status_button)
    # TODO Is there a way to reference this by name?
    loc_dropdown = button_box.children[0]
    rso_box = make_rso_buttons(original_ville, summary_box, loc_box,
                               table_box, loc_dropdown, table_dropdown, status_button)

    rso_box.children += (status_button, )
    vbox = widgets.VBox(children=[rso_box, summary_box, button_box, loc_box,
                                  org_table_box])
    return vbox

def set_up_dataset(ds):
    """Set up the dataset with all arrays we need.

    The site activation we display is `today's` activation (time.isel(0)).
    The original is the dataset as it was loaded in. The user can propose
    modifications, which are then compared to the original.

     Args:
        ds: An xr.DataSet representing the original ville to update
    """
    # Modify in place
    ds['population'] = ds['population'].astype(np.int)
    ds['original_activation'] = ds['site_activation'].copy(deep=True)
    ds['original_participants'] = ds['participants']
    ds['original_control_arm_events'] = ds['control_arm_events']
    ds['original_events'] = ds['original_control_arm_events'].sum('time').mean('scenario')
    ds['proposed_events'] = ds['control_arm_events'].sum('time').mean('scenario')
    if 'subregion1_name' in ds.data_vars: ds['SR1'] = ds['subregion1_name']
    if 'subregion2_name' in ds.data_vars: ds['SR2'] = ds['subregion2_name']
    return

def disp_table(ds, box, label_to_sort_by, num_rows=10,
               var_to_disp='population_fraction'):
    """Make the table, sort by label_to_sort_by, display in box.

    Makes a table displaying population fraction, population, current
    site_activation, proposed_events, and original_events for all locations
    in our ville.

    Args:
        ds: An xr.Dataset containing the ville to visualize.
        box: An ipywidgets.Box containing two outputs.
        label_to_sort_by: A string representing the data column to sort by.
        num_rows: An int representing the number of rows to display
        var_to_disp: A string representing the ds.data_var to show. Must
            have dims (location, *participant_dims). Canonical examples are
            'participant_fraction' and 'population_fraction'.
    """
    ds['frac_cap'] = ds['site_activation'].isel(time=0)
    ds_first = ds[[item for item in ['SR1', 'SR2', 'frac_cap', 'population'] if item in ds.data_vars]]
    ds_last = ds[['proposed_events', 'original_events']]

    int_utils.update_disp(box.children[0], var_to_disp)
    da_participants = ds[var_to_disp]
    table = table_utils.sort_table(ds_first, da_participants, ds_last, label_to_sort_by)
    styled_table = table[:num_rows].style.format({'population': '{:,}',
                                                  'frac_cap': '{:.1f}x',
                                                  'proposed_events': '{:.2f}',
                                                  'original_events': '{:.2f}'}).format(lambda x: '{:2.0f}%'.format(100*x), subset=plot_utils.get_labels_to_plot(da_participants))
    int_utils.update_disp(box.children[1], styled_table)
    return

def make_table_button(ds, box, status_button):
    """Make a widget to interactively sort the table.

    Args:
        ds: An xr.Dataset containing the ville to visualize.
        box: An ipywidgets.Box containing two outputs.
        status_button: An ipywidgets.Button to indicate when code is running
    Return:
        table_dropdown: An ipywidgets.Dropdown to sort the table.
    """
    def update_table_by_button(ds, box, status_button, label_button):
        int_utils.set_status(status_button, 'Not_Ready')
        label_to_sort_by = label_button['new']
        disp_table(ds, box, label_to_sort_by)
        int_utils.set_status(status_button, 'Ready')
    partial_disp = functools.partial(update_table_by_button, ds, box, status_button)
    label_opts = plot_utils.get_labels_to_plot(ds.participants)
    label_opts += ['population', 'proposed_events', 'original_events']
    table_dropdown = int_utils.new_dropdown(label_opts, 'Sort by:')
    table_dropdown.observe(partial_disp, type='change', names='value')
    return table_dropdown

def summary_plots(ds, box, efficacies=(0.55, 0.75)):
    """Make plots summarizing total recruitment and time to sucess.

    These plots sum information across all the ville locations. Plots the total
    number of recruits in each subgroup and the expected time to success for
    assumed vaccine efficacies.

    Args:
        ds: An xr.Dataset containing the ville to visualize.
        box: An ipywidgets.Box containing two outputs.
        efficacies: A tuple of floats representing the various assumed, vaccine
            efficacies as a percent.
    """
    # TODO add hist_events and hist_recruits

    pc = colors_config.ville_styles['gray_ville_3']['color']
    oc = colors_config.ville_styles['highlight_ville_2']['color']
    ls = '-'

    # Difference in recruits by participant label
    fig, a = plot_utils.make_subplots(1, 2, (5, 5.5), sharex=False)
    fig.suptitle('Recruitment', fontsize=16.)

    # TODO: investigate grids
    a[0].text(-0.55, 1.25, 'Trial Simulation - All Sites',
              ha='left', va='bottom', transform=a[0].transAxes,
              fontsize=20.)
    a[0].text(-0.55, 1.24, 'Compares original and proposed trial simulations',
              ha='left', va='top', transform=a[0].transAxes,
              fontsize=12.)

    proposed_unpack = plot_utils.unpack_participant_labels(ds.participants)
    original_unpack = plot_utils.unpack_participant_labels(ds.original_participants)
    labels_to_plot = plot_utils.get_labels_to_plot(ds.participants)

    plot.recruit_diffs('participant_label', a[0], proposed_unpack.sel(participant_label=labels_to_plot),
                       original_unpack.sel(participant_label=labels_to_plot), True)
    # for visibiliy, turn spines off
    plot.turn_spines_off(a[0])
    # To draw attention to xlabels, make them a little bigger
    a[0].tick_params(axis='x', labelsize=12.5)
    a[0].set_title('Proposed - original recruits')

    # Total recruits
    plot.recruits('participant_label', a[1], original_unpack.sel(participant_label=labels_to_plot),
                  oc, ls, label='original')
    plot.recruits('participant_label', a[1], proposed_unpack.sel(participant_label=labels_to_plot),
                  pc, ls, label='proposed')
    # for visibiliy, turn spines off
    a[1].set_title('Total participants')
    fig.legend(bbox_to_anchor=(1.06, 1.125))

    int_utils.update_disp(box.children[0], fig)

    # Time to success
    num_cols = len(efficacies)
    num_rows = 2
    num_plots = num_cols * num_rows
    fig, a = plot_utils.make_subplots(num_rows, num_cols, (10, 6.0), sharey='row')
    a[0].text(0.0, 1.25, 'Success day probability distribution',
              horizontalalignment='left', transform=a[0].transAxes,
              fontsize=16.)
    a[num_rows].text(0.0, 1.0, 'Proposed - original success day difference',
              ha='left', va='bottom', transform=a[num_rows].transAxes,
              fontsize=16.)

    for i, efficacy in enumerate(efficacies):
        ax = a[i]
        # tts
        plot.tts(ax, ds.control_arm_events, efficacy, pc, ls)
        plot.tts(ax, ds.original_control_arm_events, efficacy, oc, ls)
        ax.set_title(f'{efficacy} Efficacy')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)

        ax = a[i + num_cols]
        plot.turn_spines_off(ax)
        ax.tick_params(axis='y', labelsize=12.5)
        plot.tts_diff(ax,
                      ds.control_arm_events,
                      ds.original_control_arm_events,
                      efficacy)


    int_utils.update_disp(box.children[1], fig)

def loc_plots(ds, box, loc_to_plot=None):
    """Make plots summarizing incidence and recruitment in one location.

    Show data specific to one location. Plot the incidence, cumulative
    recruits, and recuits in each subgroup as functions of time.

    Args:
        ds: An xr.dataset containing the vill' to visualize.
        box: An ipywidgets.Box containing three outputs.
        loc_to_plot: A ds.location.coord specifiying the location to plot
    """
    # setup
    if loc_to_plot is None:
        loc_to_plot = ds.location.values[0]

    fpd = ville_config.FIRST_PLOT_DAY

    pc = colors_config.ville_styles['gray_ville_3']['color']
    oc = colors_config.ville_styles['highlight_ville_2']['color']
    ls = '-'

    # select just what we want to look at
    proposed_part = ds.participants.sel(location=loc_to_plot)
    original_part = ds.original_participants.sel(location=loc_to_plot)
    incidence = ds.incidence_flattened.sel(location=loc_to_plot)
    # TODO add hist_incidence and hist_recruits

    # Make plots

    # Incidence
    # No original as the user cannot interactively control the incidence.
    fig, axis = plot_utils.new_figure()

    plot.incidence(axis, incidence, fpd, 'k', '-')
    plot.format_time_axis(axis, date_format='%b-%d')
    # TODO find a better way to align plots
    axis.text(-0.25, 1.1, f'Individual Trial Site \n {loc_to_plot}',
              ha='left', va='bottom', transform=axis.transAxes,
              fontsize=16.)
    int_utils.update_disp(box.children[0], fig)

    # Total recruits over time
    fig, axis = plot_utils.new_figure()
    # TODO figure out grids
    axis.text(0.5, 1.1, ' ',
              ha='left', va='bottom', transform=axis.transAxes,
              fontsize=16.)
    # Assume everything but time is a participant dimension.
    # The arrays aren't unpacked, so this is not equivalent to
    # get_labels_to_plot.
    p_dims = list(proposed_part.dims)
    remove_dims = [plot_utils.find_time_dim(proposed_part)]
    for rd in remove_dims: p_dims.remove(rd)

    plot.cum_recruits(axis, original_part.sum(p_dims), fpd, oc, ls)
    plot.cum_recruits(axis, proposed_part.sum(p_dims), fpd, pc, ls)
    plot.format_time_axis(axis, date_format='%b-%d')
    axis.set_title('Cumulative Recruits \n All Participants')

    int_utils.update_disp(box.children[1], fig)

    # Subrecruits over time
    num_labels = len(plot_utils.get_labels_to_plot(proposed_part))
    num_cols = 2
    num_rows =  num_labels // num_cols + (num_labels % num_cols != 0)
    fig, a = plot_utils.make_subplots(num_rows, num_cols)
    fig.suptitle('Cumulative Recruits')
    plot.cum_subrecruits(a, original_part, fpd, oc, ls)
    plot.cum_subrecruits(a, proposed_part, fpd, pc, ls)

    int_utils.update_disp(box.children[2], fig)

def redraw_all(ds, sum_box, loc_box, table_box, loc_to_plot, label):
    """Update all the displays.

    Args:
        ds: An xr.dataset containing the ville to visualize.
        sum_box: An ipywidgets.Box containing the summary plots.
        loc_box: An ipywidgets.Box containing the location plots
        table_box: An ipywidgets.Box containing the table displays.
        loc_to_plot: A ds.coords['location'] indicating which location to
            display.
        label: A label to sort the table by.
    """
    disp_table(ds, table_box, label)
    summary_plots(ds, sum_box)
    loc_plots(ds, loc_box, loc_to_plot)

def update_recruitment(ds, new_participants, new_events):
    """Update recruitment DataArrays to new values.

    Args:
        ds: xr.Dataset to update.
        new_participants: xr.DataArray representing new participants.
            With dimensions (location, time, *participant_dims).
        new_events: xr.DataArray representing new control arm cum_events.
            With dimensions (location, time, *participant_dims).
    """
    ds['participants'] = new_participants
    ds['control_arm_events'] = new_events
    ds['proposed_events'] = new_events.sum('time').mean('scenario')
    return

def change_activation(new_activation, loc_to_update, ds):
    """Change the *future* site activation at a location to a new value.

    Calculate the new simulated recruitment and control arm events.

    Args:
        new_activation: Float representing the percent change to the site
            activation.
        location_to_update: A ds.coords['location'] representing the location
            to update.
        ds: An xr.Dataset to update.
        update_fn: A partial function that updates the display.
    Returns:
        None: updates ds in place
    """
    loc_arg = ds.coords['location'].values == loc_to_update

    # updating in place
    # changes for all time (not historical time)
    ds.site_activation[loc_arg] = new_activation
    participants = sim.recruitment(ds)
    events = sim.control_arm_events(ds, participants, ds.incidence_scenarios,
                                    keep_location=True)
    update_recruitment(ds, participants, events)
    return

def make_loc_buttons(ds, sum_box, loc_box, t_box, t_dropdown, status_button,
                     activation_options=(0.0, 0.5, 1.0, 1.5)):
    """Make buttons to view and modify data and activation in one location.

    Args:
        ds: An xr.dataset containing the vill' to visualize.
        sum_box: An ipywidgets.Box containing the summary plots.
        loc_box: An ipywidgets.Box containing the location plots
        status_button: An ipywidgets.Button to indicate when code is running
        activation_options: An iterable of floats representing the different
            percent activations for the site.
    Returns:
        button_box: An ipywidgets Box containing a location selection
            dropdown as well as activation buttons.
    """
    location_dropdown = int_utils.new_dropdown(sorted(list(ds.coords['location'].values)),
                                               'Location to update.')

    # Connect button to functions to call when clicked
    def loc_plots_by_button(ds, loc_box, t_box, t_dropdown, status_button,
                            loc_dropdown):
        """Wrapper function to loc_plots, with a button argument."""
        int_utils.set_status(status_button, 'Not_Ready')
        loc_to_plot = loc_dropdown['new']
        label = t_dropdown.value
        loc_plots(ds, loc_box, loc_to_plot)
        disp_table(ds, t_box, label)
        int_utils.set_status(status_button, 'Ready')
        return

    partial_loc_plots = functools.partial(loc_plots_by_button, ds, loc_box,
                                          t_box, t_dropdown, status_button)
    location_dropdown.observe(partial_loc_plots, type='change', names='value')

    # Create site activation update function
    def update_activation_by_button(loc_dropdown, ds, sum_box, loc_box, t_box,
                                    t_dropdown, status_button, new_activation,
                                    button):
        """Wrapper function to update all graphs, with a button argument."""
        # button is useless

        int_utils.set_status(status_button, 'Not_Ready')
        # Query the value here, so it will update with the dropdown selection
        loc_to_update = loc_dropdown.value
        label = t_dropdown.value
        change_activation(new_activation, loc_to_update, ds)
        redraw_all(ds, sum_box, loc_box, t_box, loc_to_update, label)
        int_utils.set_status(status_button, 'Ready')
        return

    partial_update_act = functools.partial(update_activation_by_button,
                                           location_dropdown,
                                           ds, sum_box, loc_box,
                                           t_box, t_dropdown, status_button)
    button_list = []
    for activation in activation_options:
        new_button = int_utils.new_button(f'{activation}x',
                                          f'Set activation to {activation} of capacity')
        button_fn = functools.partial(partial_update_act, activation)
        new_button.on_click(button_fn)
        button_list.append(new_button)

    button_box = widgets.HBox(children=[location_dropdown, *button_list])
    return button_box

def reset_activation(ds):
    """Reset the site activation at a location to the original values.

    Args:
        ds: An xr.Dataset to reset.
    """
    participants = ds.original_participants
    events = ds.original_control_arm_events
    ds['site_activation'] = ds.original_activation.copy(deep=True)
    update_recruitment(ds, participants, events)
    return

def save_activation(ds):
    """Write current activation to disk.

    Filepath and file open fn are defined in ville_config.

    #TODO: Give each ville a name attr?

    Args:
        ds: The xr.Dataset to save.
    """
    file_open_fn = ville_config.FILE_OPEN_FN
    # autogenerate the file_name
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f'metis_{timestamp}.nc'
    file_path = ville_config.SAVE_FILE_PATH
    # Make dir if it doesn't exist
    pathlib.Path(file_path).mkdir(exist_ok=True)
    metis.io.write_ville_to_netcdf(ds, os.path.join(file_path, file_name),
                                   file_open_fn)

def make_rso_buttons(ds, sum_box, loc_box, t_box, loc_dropdown, t_dropdown, status_button):
    """Make buttons to reset, save, and optimize site activations.

    Args:
        ds: An xr.dataset containing the ville to visualize.
        sum_box: An ipywidgets.Box containing the summary plots.
        loc_box: An ipywidgets.Box containing the location plots
        t_box: An ipywidgets.Box containing the table
        loc_dropdown: An ipywidgets.Dropdown whose value is the location
            to plot.
        t_dropdown: An ipywidgets.Dropdown whose value is the sort column for
            the table.
    Returns:
        rso_box: An ipywidgets Box containing a buttons to reset,
            and save the activations.
    """
    reset_button = int_utils.new_button('Reset', 'Reset to original, will overwrite proposed.')
    save_button = int_utils.new_button('Save', 'Save to disk.')

    def reset_redraw(ds, sum_box, loc_box, t_box, loc_dropdown, t_dropdown,
                     status_button, reset_button):
        """Wrapper to reset activations to original and redraw plots."""
        int_utils.set_status(status_button, 'Not_Ready')
        loc_to_update = loc_dropdown.value
        label = t_dropdown.value
        reset_activation(ds)
        redraw_all(ds, sum_box, loc_box, t_box, loc_to_update, label)
        int_utils.set_status(status_button, 'Ready')
        return

    def save_fn(ds, status_button, save_button):
        """Wrapper to save activations."""
        int_utils.set_status(status_button, 'Not_Ready')
        save_activation(ds)
        int_utils.set_status(status_button, 'Ready')
        return

    # Connect to buttons
    partial_reset = functools.partial(reset_redraw, ds, sum_box, loc_box, t_box,
                                      loc_dropdown, t_dropdown, status_button)
    reset_button.on_click(partial_reset)

    partial_save = functools.partial(save_fn, ds, status_button)
    save_button.on_click(partial_save)

    rso_box = widgets.HBox(children=[reset_button, save_button])
    return rso_box
