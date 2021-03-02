# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Utils to implement sortable tables."""

import pandas as pd
from bsst import plot_utils

def sort_table(ds_first, da_participants, ds_last, sort_col):
    """Sort the table and return as a pd.DataFrame indexed by location.

    We return the sorted table as a pd.DataFrame, as the pd display interface
    works well with ipywidgets.

    Args:
        ds_first: An ordered xr.DataSet, whose data_vars have dims (location,),
            to display on the left of the table.
        da_participants: An ordered xr.DataArray with dims (location, *participant_dims)
            representing some measure of the fraction of participants with different labels.
            Canonical examples are `participant_fraction` or `population_fraction`.
        ds_last: An ordered xr.DataSet, whose data_vars have dims (location,),
            to display on the right of the table.
        sort_col: A string representing the column name to sort the rows by.

    Returns:
        col_sorted_df: A pd.DataFrame with ordered columns
    """
    p_fraction = plot_utils.unpack_participant_labels(da_participants)
    p_fraction.name = p_fraction.p.item()
    del p_fraction['p']
    p_df = p_fraction.to_dataframe()
    pivot_p_df = p_df.reset_index('participant_label').pivot(columns='participant_label')
    pivot_p_df.columns = pivot_p_df.columns.droplevel()

    labels_to_keep = plot_utils.get_labels_to_plot(da_participants)
    if sort_col in labels_to_keep:
        # move to front
        labels_to_keep.remove(sort_col)
        labels_to_keep.insert(0, sort_col)
    sorted_p_df = pivot_p_df[labels_to_keep]

    # Other variables that don't depend on participant label
    first_df = ds_first.to_dataframe()

    concat_df = pd.concat((first_df, sorted_p_df), axis=1)

    if ds_last is not None:
        last_df = ds_last.to_dataframe()
        concat_df = pd.concat((concat_df, last_df), axis=1)

    sort_df = concat_df.sort_values(by=sort_col, ascending=False, na_position='last')
    return sort_df.round(decimals=2)
