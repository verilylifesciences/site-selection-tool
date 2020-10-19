# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Functions for evaluating the success of a trial."""

import numpy as np
import xarray as xr


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
  control_arm_events = control_arm_events.cumsum('time')
  time_axis = control_arm_events.dims.index('time')
  success_idx = searchsorted(
      control_arm_events.values, needed_events, axis=time_axis)
  time = control_arm_events.time.values
  time_unit = time[1] - time[0]
  success_date = time[0] + success_idx * time_unit
  return success_date
