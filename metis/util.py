# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""General purpose utilities for metis."""

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr


def expand_new_dim(dataset, name, value):
  return dataset.assign_coords({name: value}).expand_dims(name)


def xr_stack(new_dim, ds_val_pairs):
  """Like np.stack for xr.DataArrays and xr.Datasets.

  Args:
    new_dim: str, the name of the new dimension.
    ds_val_pairs: a list of (xr.Dataset, value) pairs. The datasets are assumed
      to all have the same dimensions, coordinates, and data variables.

  Returns:
    A "stacked" dataset with one additional dimension. The input dataset which
    was paired with `value` is recoverable as ds.sel(new_dim=value).
  """

  to_concat = [expand_new_dim(ds, new_dim, value) for ds, value in ds_val_pairs]
  return xr.concat(to_concat, new_dim)


def sum_all_but_dims(dims_to_keep, da):
  """Sum a xr.DataArray along all dimensions except those specified."""
  dims_to_sum = [dim for dim in da.dims if dim not in dims_to_keep]
  return da.sum(dim=dims_to_sum)


def sum_all_but_axes(axes, arr, keepdims=False):
  """Version of sum_all_but_dims for jnp.arrays."""
  num_dims = len(arr.shape)
  axes = [i % num_dims for i in axes]
  axes_to_sum = [i for i in range(num_dims) if i not in axes]
  return arr.sum(axis=axes_to_sum, keepdims=keepdims)


def searchsorted(arr, v, axis):
  """Like np.searchsorted, but supporting an axis argument."""
  v = np.asarray(v)
  arr = np.moveaxis(arr, axis, -1)
  shape = arr.shape[:-1]
  arr = arr.reshape((-1, arr.shape[-1]))
  arr = np.array([np.searchsorted(x, v) for x in arr])
  return arr.reshape(shape + v.shape)


def get_population(data: xr.Dataset) -> xr.DataArray:
  """Returns an xr.DataArray of population sizes."""
  pop = None
  if 'Population' in data.static_covariate:
    pop = data.static_covariates.sel(static_covariate='Population', drop=True)
  elif 'population' in data.static_covariate:
    pop = data.static_covariates.sel(static_covariate='population', drop=True)
  elif 'Log10Population' in data.static_covariate:
    log_population = data.static_covariates.sel(
        static_covariate='Log10Population', drop=True)
    pop = np.power(10, log_population)
  elif 'log10 population' in data.static_covariate:
    log_population = data.static_covariates.sel(
        static_covariate='log10 population', drop=True)
    pop = np.power(10, log_population)
  else:
    raise ValueError('Couldn\'t find population in static covariates '
                     f'{list(data.static_covariate)}')
  return pop.rename('population')


def needed_control_arm_events(efficacy,
                              proportion_control_arm=0.5,
                              num_stds_required=4):
  return num_stds_required**2 * (1 - efficacy * proportion_control_arm) / (
      efficacy**2 * (1 - proportion_control_arm))


def success_day(needed_events, control_arm_events):
  """Returns the first date after the given number of events is reached.

  Args:
    needed_events: a float or list/array of floats, the total number of events
      required in the control arm of the trial to declare success.
    control_arm_events: an xr.DataArray of shape [scenario, time], the number of
      control arm events on a given date (not *by* a given date).

  Returns:
    A np.array of shape [scenario] or [len(needed_events), scenario] whose
    values are the first element of control_arm_events.time after a given total
    number of control arm events is reached, or np.datetime64('NaT') if that
    number of events is not reached.
  """
  control_arm_events = control_arm_events.cumsum('time')
  time_axis = control_arm_events.dims.index('time')
  success_idx = searchsorted(
      control_arm_events.values, needed_events, axis=time_axis)
  time = control_arm_events.time.values
  time_unit = time[1] - time[0]
  success_date = time[0] + success_idx * time_unit
  did_not_succeed = success_date > control_arm_events.time.values[-1]
  if did_not_succeed.any():
    print('Warning: some scenarios did not succeed.')
    success_date = np.where(did_not_succeed, np.datetime64('NaT'), success_date)
  return success_date


def make_site_capacity_array(site_df, time, included_days_of_week=None):
  """Returns an array of daily site capacities.

  Args:
    site_df: pd.DataFrame of site information, containing a "capacity" column
      with *weekly* site capacity.
    time: xr.DataArray of times to restrict to (expected to be c.time).
    included_days_of_week: optional list of integers (0 through 6) indicating
      which days of the week sites recruit participants (Monday through Sunday,
      respectively).

  Returns:
    An xr.DataArray of shape [location, time], the number of participants a site
    can recruit on each day.
  """
  if included_days_of_week is None:
    included_days_of_week = list(range(7))
  site_capacity = site_df.capacity.to_xarray() / len(included_days_of_week)
  site_capacity = site_capacity.broadcast_like(time).astype('float')
  excluded_days = [d for d in range(7) if d not in included_days_of_week]
  is_excluded = xr.apply_ufunc(
      lambda x: pd.to_datetime(x).dayofweek.isin(excluded_days),
      site_capacity.time)
  site_capacity = xr.where(is_excluded, 0.0, site_capacity)

  # Can't recruit before the activation date
  activation_date = site_df.start_date.to_xarray()
  for l in activation_date.location.values:
    date = activation_date.loc[l]
    site_capacity.loc[site_capacity.time < date, l] = 0.0

  return site_capacity.transpose('location', 'time')


def fraction_and_incidence_scaler(site_df,
                                  dim_name,
                                  cols,
                                  scalers,
                                  default_col=None):
  """Returns fraction and incidence scaler arrays based on site data.

  Args:
    site_df: pd.DataFrame with site data. Must contain columns `cols`, where
      site_df[col] is the fraction of the population in a given category.
    dim_name: string, name of the dimension.
    cols: the columns of site_df to extract. Each column is the fraction of the
      population in a given category, and the categories are assumed to be
      non-overlapping.
    scalers: list of incidence scalers the same length as `cols`.
    default_col: optional, the name of the "catch-all other" column to add. If
      omitted, it is assumed that the provided columns add up to 1.0. The
      incidence scaler for this category will be 1.0.

  Returns:
    A pair of xr.DataArrays, the first of which is the proportion of the
    population in each category at each location (of shape [location,
    dim_name]), and the second of which is the incidence scaler for each
    category (of shape [dim_name]).
  """
  df = site_df[cols]
  if default_col is not None:
    default = (1.0 - df.sum(axis=1)).rename(default_col)
    df = pd.concat([default, df], axis=1)
    scalers = [1.] + list(scalers)
  df.columns.name = dim_name
  fraction = df.stack().to_xarray()
  scaler = xr.DataArray(scalers, coords=(fraction[dim_name],))
  return fraction, scaler


def add_empty_history(c):
  time = np.array([], dtype='datetime64')
  time = xr.DataArray(time, dims=('historical_time',))
  time = time.assign_coords(historical_time=time).astype('float')
  location = xr.zeros_like(c.location).astype('float')
  c['historical_participants'] = location * time * c.incidence_scaler
  c['historical_site_activation'] = location * time
  c['historical_incidence'] = location * time
  c['historical_control_arm_events'] = location * time


def linear_interpolation_weights(x, xnew):
  """Weight matrix satisfying np.dot(x, weights) == xnew.

  This expresses each element of xnew as a linear interpolation of the values of
  x immediately above and below it (or the two closest values, if extrapolation
  is required).

  Args:
    x: a 1-dimensional *sorted* np.array.
    xnew: a 1-dimensional np.array

  Returns:
    A 2-dimensional np.array of weights of shape (len(x), len(xnew)) satisfying
    np.dot(x, weights) == xnew
    where weights[:, i] has two consecutive non-zero entries summing to 1.
  """
  xnew_indices = np.searchsorted(x, xnew)
  hi = np.minimum(np.maximum(xnew_indices, 1), len(x) - 1)
  lo = hi - 1
  weight_hi = (xnew - x[lo]) / (x[hi] - x[lo])
  weight_lo = 1 - weight_hi
  weights = np.zeros((len(x), len(xnew)))
  weights[lo, np.arange(len(xnew))] = weight_lo
  weights[hi, np.arange(len(xnew))] = weight_hi
  return weights


def quantile_conversion_weights(quantiles, new_quantiles):
  """Weights for linearly interpolating quantiles to new_quantiles.

  Assumes values are normally distributed.

  Args:
    quantiles: a sorted 1-dimesional np.array of values between 0.0 and 1.0.
    new_quantiles: a 1-dimensional np.array of values between 0.0 and 1.0.

  Returns:
    A 2-dimensional array of weights for converting quantiles of a
    normally-distributed quantity. This weight matrix roughly satisfies
    samples = np.random.normal(size=big_number)
    x = np.quantile(samples, quantiles)
    weights = quantile_conversion_weights(quantiles, new_quantiles)
    y_pred = np.dot(x, weights)
    y_true = np.quantile(samples, new_quantiles)
    np.testing.assert_allclose(y_pred, y_true)
  """
  return linear_interpolation_weights(
      scipy.stats.norm.ppf(quantiles), scipy.stats.norm.ppf(new_quantiles))


def constant_extrapolate(p, extrap_day):
  """Constant extrapolates xr.DataArray p until extrap_day."""
  time_step = p.time.values[1] - p.time.values[0]
  extrap_time = np.arange(p.time.values[-1] + time_step,
                          extrap_day + np.timedelta64(1, 's'), time_step)
  extrap_time = xr.DataArray(extrap_time, coords=(extrap_time,), dims=('time',))
  extrap_p = p.isel(
      time=-1, drop=True) * xr.ones_like(extrap_time).astype(float)
  return xr.concat([p, extrap_p], 'time')


def reindex_by_site_id(site_df, ds):
  """Reindexes an opencovid_key-indexed dataset by site_id."""
  ds = ds.reindex(location=site_df.opencovid_key.values)
  ds['location'] = site_df.index.values
  return ds
