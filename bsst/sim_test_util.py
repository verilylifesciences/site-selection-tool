# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Constructs structures needed for testing recruitment or infection simulation."""

# Not sure why, but the attributes `dims`, `coords`, `shape` of  xr.DataArrays
# trip up pytype.
# pytype: skip-file

import numpy as np
import pandas as pd
import xarray as xr
from bsst import sim
from bsst import util


def c_to_test_recruitment(rand=None):
  """Trial dataset for testing sim.recruitment."""
  if rand is None:
    rand = np.random.RandomState(seed=0)
  sites = ['one', 'two', 'three', 'four']
  historical_time = pd.date_range('2020-09-20', '2020-10-01')
  time = pd.date_range('2020-10-02', '2020-12-01')
  site_capacity = xr.DataArray(
      20 * rand.rand(len(sites), len(time)),
      dims=('location', 'time'),
      coords=(sites, time))
  site_activation = site_capacity.copy()
  site_activation.values = rand.rand(*site_activation.shape)

  participant_fraction = xr.DataArray(
      rand.rand(len(sites), 3),
      dims=('location', 'ethnicity'),
      coords=(
          sites,
          ['other', 'black', 'hispanic'],
      ))
  participant_fraction /= util.sum_all_but_dims(['location'],
                                                participant_fraction)
  historical_participants = xr.DataArray(
      rand.rand(
          len(sites), len(historical_time,), *participant_fraction.shape[1:]),
      dims=(('location', 'historical_time') + participant_fraction.dims[1:]),
      coords=(sites, historical_time) +
      tuple(participant_fraction.coords.values())[1:])

  c = xr.Dataset()
  c['site_capacity'] = site_capacity
  c['site_activation'] = site_activation
  c['historical_participants'] = historical_participants
  c['participant_fraction'] = participant_fraction
  c['trial_size_cap'] = 200
  return c


def c_to_test_events(rand=None):
  """Trial dataset for testing sim.control_arm_events."""
  if rand is None:
    rand = np.random.RandomState(seed=0)
  sites = ['one', 'two', 'three', 'four']

  incidence_scaler = xr.DataArray([1., 2., 3.],
                                  dims=('ethnicity',),
                                  coords=(['other', 'black', 'hispanic'],))
  incidence_to_event_factor = xr.DataArray(
      [.6, .7, .8],
      dims=('ethnicity',),
      coords=(['other', 'black', 'hispanic'],))
  population_fraction = xr.DataArray(
      rand.rand(len(sites), 3),
      dims=('location', 'ethnicity'),
      coords=(
          sites,
          ['other', 'black', 'hispanic'],
      ))
  population_fraction /= util.sum_all_but_dims(['location'],
                                               population_fraction)

  c = xr.Dataset()
  c['proportion_control_arm'] = .4
  c['observation_delay'] = 7  # note this is less than historical_time above
  c['incidence_scaler'] = incidence_scaler
  c['incidence_to_event_factor'] = incidence_to_event_factor
  c['population_fraction'] = population_fraction

  return c


def participants_and_forecast(rand=None):
  """Participant and incidence arrays for testing sim.control_arm_events."""

  if rand is None:
    rand = np.random.RandomState(seed=0)
  c = c_to_test_recruitment(rand)
  participants = sim.recruitment(c)

  num_scenarios = 10
  incidence_scenarios = 0.001 * xr.DataArray(
      rand.rand(num_scenarios, c.location.size, c.time.size),
      dims=('scenario', 'location', 'time'),
      coords=(range(num_scenarios), c.location.values, c.time.values))
  return participants, incidence_scenarios
