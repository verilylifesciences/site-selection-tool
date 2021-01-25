# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Utilities for simulating recruitment and infection of trial participants."""

# I know you mean well, pytype, but I'd like my janky JaxDataset.
# pytype: skip-file

from typing import List
from flax import nn
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from metis import sim_scenarios
from metis import util


class JaxDataset:
  """Roughly a xr.Dataset with all arrays turned into jnp.arrays.

  Warning: this doesn't do any intelligent broadcasting or indexing. It just
  shoves array values into jnp.arrays.
  """

  def __init__(self, ds):
    for var_name, var in ds.data_vars.items():
      if var.dtype == np.dtype('O'):
        print(f'Skipping jax conversion of {var_name} (dtype is "object")')
        continue
      if var.dims:
        setattr(self, var_name, jnp.array(var.values))
      elif np.issubdtype(var.dtype, np.datetime64):
        setattr(self, var_name, var.values)
      else:
        setattr(self, var_name, var.item())

  def __repr__(self):
    return repr(self.__dict__)


def cumsum_capper(x, threshold):
  """A multiplier for capping the cumsum of x to the given threshold.

  Args:
    x: one-dimensional xr.DataArray with non-negative values.
    threshold: maximum desired cumsum value.

  Returns:
    An xr.DataArray with the same shape as x with the property that
    (result * x).cumsum() = where(x.cumsum() < threshold, x.cumsum(), threshold)
  """
  # The variable naming here assumes the coordinate is time and the intervals
  # are days.
  cumsum = x.cumsum()
  time = x.dims[0]
  first_idx_over = np.searchsorted(cumsum, threshold)
  if first_idx_over == len(x):
    # No capping necessary because cumsum never hits the threshold.
    return xr.ones_like(x)
  x_that_day = x.isel({time: first_idx_over})
  if first_idx_over == 0:
    x_allowed_that_day = threshold
  else:
    x_allowed_that_day = (threshold - cumsum.isel({time: first_idx_over - 1}))

  scale_that_day = x_allowed_that_day / x_that_day
  capper = xr.ones_like(x)
  capper[first_idx_over] = scale_that_day
  capper[first_idx_over + 1:] = 0.0
  return capper


def differentiable_cumsum_capper(x, threshold, width):
  """Starts out 1 and decays to 0 so that (result * x).cumsum() <= threshold."""
  cumsum = x.cumsum()
  capped_cumsum = threshold - width * nn.softplus((threshold - cumsum) / width)
  capped_x = jnp.concatenate([x[:1], jnp.diff(capped_cumsum)])
  # The "double where" trick is needed to ensure non-nan gradients here. See
  # https://github.com/google/jax/issues/1052#issuecomment-514083352
  nonzero_x = jnp.where(x == 0.0, 1.0, x)
  capper = jnp.where(x == 0.0, 0.0, capped_x / nonzero_x)
  return capper


def differentiable_greater_than(x, threshold, width):
  """A smoothed version of (x > threshold).astype(float)."""
  return nn.sigmoid((x - threshold) / width)


def recruitment_rules_simple(c: xr.Dataset) -> List[str]:
  return [f'At most {c.trial_size_cap.item()} participants can be recruited.']


def recruitment_simple(c: xr.Dataset) -> xr.DataArray:
  # pyformat: disable
  """Returns new recruited participants each day.

  Args:
    c: xr.Dataset capturing recruitment rules. This is required to have data
      variables with the following names and shapes:
      * site_activation [location, time], 0-to-1 float valued array indicating
        what proportion of its capacity each site recruits on each future day.
      * site_capacity [location, time], the maximum number of new recruits at
        each site on each future day.
      * historical_participants [location, historical_time, ...], the number of
        participants recruited at each location in each demographic bucket for
        each day in the past.
      * participant_fraction [location, ...], the proportion of recruits at
        a given location falling into given demographic buckets.
      * trial_size_cap [], maximum number of participants.

  Returns:
    An xr.DataArray of shape [location, time, ...], the number of
    participants recruited at a given location on a given day within each
    demographic bucket. This includes both control and vaccine arms. The time
    dimension of this array includes both the past and the future.
  """
  # pyformat: enable
  if c.site_activation.isnull().any():
    raise ValueError('NaNs in site_activation array!')
  if c.site_capacity.isnull().any():
    raise ValueError('NaNs in site_capacity array!')
  if c.historical_participants.isnull().any():
    raise ValueError('NaNs in historical_participants array!')
  if c.participant_fraction.isnull().any():
    raise ValueError('NaNs in participant_fraction array!')
  if c.trial_size_cap.isnull():
    raise ValueError('trial_size_cap is NaN!')

  future_participants = c.site_activation * c.site_capacity

  # Prepend historical_participants, which is assumed to start on start_day.
  historical = c.historical_participants.rename(historical_time='time')
  historical = util.sum_all_but_dims(['location', 'time'], historical)
  participants = xr.concat([historical, future_participants], dim='time')

  # After the trial size is reached globally, all locations must stop
  # recruiting.
  capper = cumsum_capper(participants.sum('location'), c.trial_size_cap)
  participants *= capper

  # Break into demographic buckets.
  fractions = (
      c.participant_fraction /
      util.sum_all_but_dims(['location'], c.participant_fraction))
  participants = participants * fractions

  # Overwrite historical participants; the fractions are wrong otherwise.
  the_future = (future_participants.time.values[0] <= participants.time)
  participants = xr.concat([
      c.historical_participants.rename(historical_time='time'),
      participants.sel(time=the_future)
  ],
                           dim='time')

  # Ensure the dimensions are ordered correctly.
  dims = ('location', 'time') + c.participant_fraction.dims[1:]
  return participants.transpose(*dims).rename('participants')


def differentiable_recruitment_simple(c: JaxDataset, width=1 / 3.):
  """Smoothed version of recruitment_simple with jnp.arrays.

  It is expected that the values of recruitment_simple(c) closely matches
  differentiable_recruitment_simple(JaxDataset(c)), especially with small width.

  Args:
    c: a JaxDataset specifying the trial. See the documentation of
      recruitment_simple for what fields are required.
    width: how sharp the smoothed step functions should be, measured
      approximately in days.

  Returns:
    A jnp.array of shape [location, time, ...], the number of participants
    recruited at a given location on a given day within each demographic bucket.
    This includes both control and vaccine arms. The time dimension of this
    array includes both the past and the future.
  """
  # [future_]participants is of shape [location, start_time]
  future_participants = c.site_activation * c.site_capacity

  # Prepend historical_participants, which is assumed to start on start_day.
  historical = c.historical_participants
  historical = util.sum_all_but_axes([0, 1], historical)
  participants = jnp.concatenate([historical, future_participants], axis=1)

  first_future_idx = historical.shape[1]
  mean_daily_recruitment = util.sum_all_but_axes([1], participants).mean()

  # After the trial size is reached globally, all locations must stop
  # recruiting.
  capper = differentiable_cumsum_capper(
      participants.sum(axis=0),
      c.trial_size_cap,
      width=mean_daily_recruitment * width)
  participants = participants * capper[jnp.newaxis, :]

  # Break into demographic buckets.
  fractions = (
      c.participant_fraction /
      util.sum_all_but_axes([0], c.participant_fraction, keepdims=True))
  fractions = fractions[:, jnp.newaxis, ...]  # add dimension for time
  # Add dimensions for demographic labels.
  for _ in range(len(c.participant_fraction.shape) - 1):
    participants = participants[..., jnp.newaxis]
  participants = participants * fractions

  # Overwrite historical participants; the fractions are wrong otherwise.
  participants = jnp.concatenate(
      [c.historical_participants, participants[:, first_future_idx:]], axis=1)

  return participants


def control_arm_events(c,
                       participants,
                       incidence_scenarios,
                       keep_location=False):
  # pyformat: disable
  """Returns numbers of control arm events over time in the future.

  Args:
    c: a xr.Dataset specifying the trial. This is required to have data
      variables with the following names and shapes:
      * proportion_control_arm [], what fraction of participants are in the
        control arm of the trial.
      * observation_delay [], how long after a participant is recruited before
        they enter the observation period.
      * incidence_scaler [...], the relative risk of a participant in a
        given demographic bucket.
      * incidence_to_event_factor [...], the proportion of incidence which
        meets the criteria to be a trial event.
      * population_fraction [location, ...], the proportion of the
        population in each demographic bucket in each location.
    participants: xr.DataArray of shape [location, time, ...], the number
      of participants recruited each day at each location in each demographic
      bucket. This includes both control and vaccine arms. The time dimension of
      this array includes both the past and the future.
    incidence_scenarios: xr.DataArray of shape [scenario, location, time], the
      fraction of the population at each location who will be infected on each
      day in the future.
    keep_location: If True, don't sum over the location dimension.

  Returns:
    A xr.DataArray of shape [scenario, time] (or [scenario, location, time] if
    keep_location is True), the number of events each day in the control arm of
    the trial in each scenario.
  """
  # pyformat: enable

  if c.proportion_control_arm.isnull():
    raise ValueError('proportion_control_arm is NaN!')
  if c.observation_delay.isnull():
    raise ValueError('observation_delay is NaN!')
  if c.incidence_scaler.isnull().any():
    raise ValueError('NaNs in incidence_scaler array!')
  if c.incidence_to_event_factor.isnull().any():
    raise ValueError('NaNs in incidence_to_event_factor array!')
  if c.population_fraction.isnull().any():
    raise ValueError('NaNs in population_fraction array!')
  if participants.isnull().any():
    raise ValueError('NaNs in participants array!')
  if incidence_scenarios.isnull().any():
    raise ValueError('NaNs in incidence_scenarios array!')

  observation_delay = int(c.observation_delay)
  if observation_delay < 0 or participants.time.size < observation_delay:
    raise ValueError(f'Observation delay ({observation_delay}) is negative or '
                     f'greater than the trial duration.')

  # Restrict to control arm participants.
  participants = participants * c.proportion_control_arm
  # Switch to start of observation instead of recruitment date.
  participants = participants.shift(time=observation_delay).fillna(0.0)

  # Treat all participants whose observation period starts on or before the
  # first day of forecast as starting on the first day of forecast.
  immediate_participants = participants.sel(
      time=(
          participants.time <= incidence_scenarios.time.values[0])).sum('time')
  immediate_participants = immediate_participants.expand_dims(
      'time').assign_coords(dict(time=incidence_scenarios.time.values[:1]))
  participants = xr.concat([
      immediate_participants,
      participants.sel(time=incidence_scenarios.time.values[1:])
  ],
                           dim='time')

  participants = participants.rename(time='observation_start')

  # Incidence for a subpopulation with incidence_scaler 1.0. Its shape is
  # [scenario, location, time].
  normalizing_constant = c.population_fraction.dot(c.incidence_scaler)
  bad_locations = list(
      normalizing_constant.location.values[(normalizing_constant == 0).values])
  if bad_locations:
    print(bad_locations)
    raise ValueError(
        'The following locations have no people in the population who can get '
        'infected (i.e. population_fraction.dot(incidence_scaler) is zero). It '
        'is impossible to account for incidence!\n' + ','.join(bad_locations))
  baseline_incidence = incidence_scenarios / normalizing_constant
  # The effective number of "unit risk" participants who start observation each
  # day. Its shape is [location, observation_start]. By dotting out the
  # dimensions having to do with demographic labels early, we avoid
  # instantiating a huge array when we multiply by baseline_incidence;
  # an array of shape [scenario, location, time, observation_start] (when
  # keep_location=True) is large enough without additional dimensions for
  # demographics.
  event_factor = c.incidence_scaler * c.incidence_to_event_factor
  effective_participants = participants.dot(event_factor)

  if keep_location:
    ctrl_arm_events = baseline_incidence * effective_participants
    final_dims = ('scenario', 'location', 'time')
  else:
    ctrl_arm_events = baseline_incidence.dot(effective_participants)
    final_dims = ('scenario', 'time')

  # You cannot have an event before the start of observation.
  ctrl_arm_events = xr.where(
      baseline_incidence.time < effective_participants.observation_start, 0.0,
      ctrl_arm_events)

  return ctrl_arm_events.sum('observation_start').transpose(*final_dims)


def shift_pad_zeros(arr, shift, axis):
  """Like jnp.roll, but fills with zeros instead of wrapping around."""
  axis = axis % len(arr.shape)  # to handle things like axis=-1
  sl = (slice(None),) * axis
  zeros = jnp.zeros_like(arr[sl + (slice(0, shift),)])
  trim_last_n = slice(None, arr.shape[axis] - shift)
  return jnp.concatenate([zeros, arr[sl + (trim_last_n,)]], axis=axis)


def differentiable_control_arm_events(c, participants, incidence_scenarios):
  """Smoothed version of control_arm_events with jnp.arrays.

  Args:
    c: a jax-ified TrialConfig specifying the trial.
    participants: jnp.array of shape [location, time, age, ...], the number of
      participants recruited each day at each location in each demographic
      bucket. This includes both control and vaccine arms.
    incidence_scenarios: jnp.array of shape [scenario, location, time], the
      fraction of the population at each location who will be infected on each
      day.

  Returns:
    A jnp.array of shape [scenario, time], the number of events each day in the
    control arm of the trial in each scenario.
  """
  # Restrict to control arm participants.
  participants = participants * c.proportion_control_arm
  # Switch from recruitment date to the start of observation. The shape of
  # participants is now [location, observation_start, age, ...].
  participants = shift_pad_zeros(participants, c.observation_delay, axis=1)

  # Treat all participants whose observation period starts on or before the
  # first day of forecast as starting on the first day of forecast.
  second_day_of_forecast = -incidence_scenarios.shape[2] + 1
  immediate_participants = participants[:, :second_day_of_forecast].sum(
      axis=1, keepdims=True)
  participants = jnp.concatenate(
      [immediate_participants, participants[:, second_day_of_forecast:]],
      axis=1)

  # Incidence for a subpopulation with incidence_scaler 1.0. Its shape is
  # [scenario, location, time].
  normalizing_constant = jnp.einsum('l...,...->l', c.population_fraction,
                                    c.incidence_scaler)
  normalizing_constant = normalizing_constant[jnp.newaxis, :, jnp.newaxis]
  baseline_incidence = incidence_scenarios / normalizing_constant
  # The effective number of "unit risk" participants who start observation each
  # day. Its shape is [location, observation_start].
  event_factor = c.incidence_scaler * c.incidence_to_event_factor
  effective_participants = jnp.einsum('lo...,...->lo', participants,
                                      event_factor)

  ctrl_arm_events = jnp.einsum('slt,lo->sto', baseline_incidence,
                               effective_participants)

  # You cannot have an event before the start of observation.
  t_less_than_o = (slice(None),) + jnp.triu_indices_from(
      ctrl_arm_events[0], k=1)
  ctrl_arm_events = jax.ops.index_update(ctrl_arm_events, t_less_than_o, 0.0)

  return ctrl_arm_events.sum(axis=-1)


RECRUITMENT_REGISTRY = dict()


def register_recruitment_type(name, recruitment_fn, differentiable_fn,
                              rules_fn):
  """Register a recruitment scheme.

  Args:
    name: a string to identify the recruitment scheme.
    recruitment_fn: a function which takes an xr.Dataset specifying the trial
      and returns an xr.DataArray of recruited participants.
    differentiable_fn: a differentiable version of recruitment_fn which takes a
      jax-ified trial dataset and returns a jnp.array. This function is assumed
      to also have an optional `width` argument specifying the amount of
      smoothing. If width is made close to 0.0, differentiable_fn(JaxDataset(c),
      width) should be very close to recruitment_fn(c).values.
    rules_fn: a function which takes an xr.Dataset specifying the trial and
      returns a list of human-readable descriptions of what rules are used for
      recruitment.
  """
  RECRUITMENT_REGISTRY[name] = (recruitment_fn, differentiable_fn, rules_fn)


register_recruitment_type('default', recruitment_simple,
                          differentiable_recruitment_simple,
                          recruitment_rules_simple)


def get_recruitment_type(c):
  if 'recruitment_type' in c:
    return c.recruitment_type.item()
  return 'default'


def recruitment(c: xr.Dataset) -> xr.DataArray:
  recruitment_fn, _, _ = RECRUITMENT_REGISTRY[get_recruitment_type(c)]
  return recruitment_fn(c)


def recruitment_rules(c: xr.Dataset):
  _, _, rules_fn = RECRUITMENT_REGISTRY[get_recruitment_type(c)]
  return rules_fn(c)


def add_stuff_to_ville(c, incidence_model, site_df, num_scenarios=300):
  """Adds incidence and site data we've deemed important to a trial dataset.

  Args:
    c: xr.Dataset, a trial config suitable for running recruitment and event
      simulation.
    incidence_model: xr.DataArray of shape [model, sample, location, time], the
      forecast incidence.
    site_df: pd.Dataframe with information about sites.
    num_scenarios: the number of scenarios to generate.
  """
  included_days = np.logical_and(c.time.values[0] <= incidence_model.time,
                                 incidence_model.time <= c.time.values[-1])
  incidence_model = incidence_model.sel(time=included_days)
  c['incidence_model'] = incidence_model
  c['incidence_flattened'] = sim_scenarios.get_incidence_flattened(
      c.incidence_model, c)
  c['incidence_scenarios'] = sim_scenarios.generate_scenarios_independently(
      c.incidence_flattened, num_scenarios)

  if 'participants' not in c.data_vars:
    print('Populating participants.')
    participants = recruitment(c)
    c['participants'] = participants.sel(time=c.time)
  else:
    participants = c.participants
  if 'control_arm_events' not in c.data_vars:
    print('Populating control_arm_events based on independent scenarios.')
    ctrl_arm_events = control_arm_events(
        c, participants, c.incidence_scenarios, keep_location=True)
    c['control_arm_events'] = ctrl_arm_events

  site_fields = [
      'subregion1_name',
      'subregion2_name',
      'address',
      'lat',
      'lon',
      'opencovid_key',
      'population',
      'site_name',
      'site_id',
      'full_population',
  ]
  for f in site_fields:
    if f in site_df.columns:
      c[f] = site_df[f].to_xarray()
