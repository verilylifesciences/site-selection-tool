# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd
"""Utilities for making and interacting with ipywidgets."""

import IPython.display as dsp
import ipywidgets as widgets

def update_disp(widget, fig):
    """Update the widget with a figure.

    Args:
        widget: An ipywidget to update
        fig: A figure to display
    """
    with widget:
        dsp.clear_output(wait=True)
        dsp.display(fig)

def new_dropdown(options, description):
    """Make a new dropdown widget.

    Args:
        options: An iterable of options to choose between.
        description: A string representing the mouse-over description
    Returns:
        widget: A dropdown widget to choose between options.
    """
    widget = widgets.Dropdown(
    options=options,
    value=options[0],
    rows=10,
    description=description,
    disabled=False,
    style={'description_width': 'initial'}
    )
    return widget

def new_button(text, description):
    """Make a new button widget.

    Args:
        text: A string to display on the button.
        description: A string representing the mouse-over description
    Returns:
        widget: A button widget.
    """
    widget = widgets.Button(
        description=text,
        disabled=False,
        button_style='',
        tooltip=description
    )
    return widget

def new_status_button():
    """Make a new button widget to show when the notebook is processing data.

    Returns:
        widget: A button widget.
    """
    indicator = new_button('', 'Status indicator')
    set_status(indicator, 'Ready')
    return indicator

def set_status(status_button, new_status='False'):
    """Toggle the display of the status button.

    Args:
        status_button: An widgets.Button
        new_status: A string representing the new system status. 'Ready' is
            interpreted as 'change color to green.' (Almost) anything is
            interpreted as 'change color to pink.'
    """
    if new_status in ['Ready', 'r', 'ready', 'R']:
        status_button.style.button_color = 'greenyellow'
        status_button.icon = 'check'
    else:
        status_button.style.button_color='hotpink'
        status_button.icon = 'ellipsis-h'
    return
