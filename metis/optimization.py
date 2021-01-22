# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Optimization of vaccine trials."""

from absl import logging
from flax import nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

import metis.sim as sim


class Parameterizer:
  """An interface for parameterizing whatever the trial planner can control."""

  def init_params(self):
    """Returns an xr.DataArray of initial parameters of the right shape."""
    raise NotImplementedError

  def apply_params(self, c: sim.JaxDataset, params: jnp.array):
    """Updates a jax-ified trial in place with the given parameters.

    This method must be differentiable, as it's used for optimization.

    Args:
      c: a jax-ified dataset to be modified in place.
      params: a jnp.array of parameters to be applied to c.
    """
    raise NotImplementedError

  def xr_apply_params(self, c: xr.Dataset, params: xr.DataArray):
    """Like apply_params, but with xarray instead of jax."""
    c_ = sim.JaxDataset(c)
    self.apply_params(c_, params.values)
    # Note: We assume here that site_activation is the only thing being wiggled
    # by apply_params. If you write a Parameterizer which messes with other
    # fields, those values should be copied over too.
    c.site_activation.values = c_.site_activation


class StaticSiteActivation(Parameterizer):
  """A Parameterizer which allows the planner to choose fixed site activations.

  There's one parameter per location, sigmoid of which is the location's
  activation level.
  """

  def __init__(self, c):
    self.location = c.location

  def init_params(self):
    return xr.DataArray(
        np.random.normal(size=self.location.size), coords=(self.location,))

  def apply_params(self, c, params):
    params = jnp.broadcast_to(params[:, jnp.newaxis], c.site_capacity.shape)
    c.site_activation = nn.sigmoid(params)


class DynamicSiteActivation(Parameterizer):
  """A Parameterizer which allows the planner to change site activation freely.

  There's one parameter per location and day, sigmoid of which is the location's
  activation level on that day.
  """

  def __init__(self, c):
    self.location = c.location
    self.time = c.time

  def init_params(self):
    return xr.DataArray(
        np.random.normal(size=(self.location.size, self.time.size)),
        coords=(self.location, self.time))

  def apply_params(self, c, params):
    c.site_activation = nn.sigmoid(params)


class PivotTableActivation(Parameterizer):
  """A class for capping the number of sites activated from certain groups.

  Each site is assigned to a group (e.g. US vs ex-US), specified by the
  `site_to_group` series. We suppose there are several dates on which site
  activations are allowed to change, and that a given number of sites from each
  group can be activated. The [group, decision_day]-shaped dataframe
  `allowed_activations` specifies the a cap on how much site activation can be
  *increased* on any given decision date. If can_deactivate is true, there is no
  limit to how much site activation can be *decreased* (aside from the limit
  where everything is turned off).

  Notes:
    * When can_deactivate is True, deactivating a site does not give you "extra
      credits" which you can use to activate more sites.
    * If you later decide you want to re-activate a deactivated site again,
      it'll cost you credits, since that's an increase in activation.
    * Any locations not in a group are unconstrained. They can be turned on or
      off "for free" on each decision day.
  """

  def __init__(self,
               c: xr.Dataset,
               site_to_group: pd.Series,
               allowed_activations: pd.DataFrame,
               force_hit_cap: bool = False,
               can_deactivate: bool = True):
    groups = allowed_activations.index.values
    self.site_to_group = site_to_group.loc[c.location.values]
    given_decision_days = allowed_activations.columns.values
    self.decision_day_idx = np.searchsorted(c.time.values, given_decision_days)
    self.group_to_idx = {
        group: np.where(self.site_to_group == group)[0] for group in groups
    }
    self.allowed_activations = allowed_activations.copy()
    used_decision_days = c.time.values[self.decision_day_idx]
    self.allowed_activations.columns = used_decision_days
    if not (used_decision_days == given_decision_days).all():
      logging.warning(
          'Some decision dates not found in c.time. Using decision dates\n%s\n'
          'instead of\n%s', used_decision_days, given_decision_days)

    self.can_deactivate = can_deactivate
    self.force_hit_cap = force_hit_cap
    if force_hit_cap and can_deactivate:
      raise ValueError(
          'The combination force_hit_cap=True and can_deactivate=True is not '
          'supported. The "squash up" logic to enforce hitting the cap gets '
          'annoying because it could require turning negative activations into '
          'positive activations.')

  def init_params(self):
    decision_days = self.allowed_activations.columns.values
    locations = self.site_to_group.index.values
    params = xr.DataArray(
        np.random.normal(size=(len(decision_days), len(locations))),
        dims=('decision_day', 'location'),
        coords=(decision_days, locations))
    return params

  def apply_params(self, c, params):
    # Make an activation array of shape [decision_day, location] which satisfies
    # the pivot table constraint.
    site_activation = nn.sigmoid(params)
    for i in range(len(self.decision_day_idx)):
      new_activation, persistence_and_deactivation = (
          self._new_and_old_activation(site_activation[:i + 1]))
      for group, group_idx in self.group_to_idx.items():
        num_allowed = self.allowed_activations.loc[group].iloc[i]
        num_activated = new_activation[group_idx].sum()
        if num_activated > num_allowed:
          squashed = (num_allowed / num_activated) * new_activation[group_idx]
          new_activation = jax.ops.index_update(new_activation,
                                                jax.ops.index[group_idx],
                                                squashed)
        elif self.force_hit_cap:
          # Same idea as above, but squash towards 1 instead of towards 0.
          max_new_possible = 1 - persistence_and_deactivation[group_idx]
          excess_available = max_new_possible - new_activation[group_idx]
          total_excess_available = excess_available.sum()
          total_excess_required = num_allowed - num_activated
          scaler = 1.0
          if total_excess_available > total_excess_required:
            scaler = total_excess_required / total_excess_available
          squashed = new_activation[group_idx] + scaler * excess_available
          new_activation = jax.ops.index_update(new_activation,
                                                jax.ops.index[group_idx],
                                                squashed)
      adjusted_activation = persistence_and_deactivation + new_activation
      site_activation = jax.ops.index_update(site_activation,
                                             jax.ops.index[i, :],
                                             adjusted_activation)

    # Expand from [decision_day, location] to [time, location].
    c.site_activation = jnp.zeros(c.site_activation.shape)
    for i in range(len(self.decision_day_idx) - 1):
      day_idx = self.decision_day_idx[i]
      next_day_idx = self.decision_day_idx[i + 1]
      c.site_activation = jax.ops.index_update(
          c.site_activation, jax.ops.index[:, day_idx:next_day_idx],
          site_activation[i, :, jnp.newaxis])
    c.site_activation = jax.ops.index_update(
        c.site_activation, jax.ops.index[:, self.decision_day_idx[-1]:],
        site_activation[-1, :, jnp.newaxis])

  def _new_and_old_activation(self, site_activation: jnp.array):
    """Returns new activation on the most recent day, and everything else.

    Args:
      site_activation: jnp.array of shape [decision_day, location], the
        activation of each site on each decision day.

    Returns:
      A pair of jnp.arrays of the same shape as site_activation[-1]. The first
      is the new activation of each site on the last decision day (i.e. how much
      more activated it is relative to the previous decision day, or zero if it
      is not more active). The second is everything else, so the sum of the two
      is equal to site_activation[-1] (or if can_deactivate is False,
      maximum(*site_activation[-2, -1]) to reflect the impossibility of
      deactivation).
    """
    # Would softening the maximum to allow gradients to flow help?
    new_activation = site_activation[-1]
    if len(site_activation) > 1:
      new_activation = new_activation - site_activation[-2]
    new_activation = jnp.maximum(new_activation, 0.0)
    total = site_activation[-1]
    if not self.can_deactivate and len(site_activation) > 1:
      total = jnp.maximum(total, site_activation[-2])
    persistence_and_deactivation = total - new_activation
    return new_activation, persistence_and_deactivation

  def greedy_activation(self, c: xr.Dataset,
                        incidence_scenarios: xr.DataArray) -> xr.DataArray:
    """The site activation plan which turns sites on in order of incidence."""
    site_activation = xr.zeros_like(c.site_activation)
    already_activated = []
    obs_delay = np.timedelta64(c.observation_delay.item(), 'D')
    for date in self.allowed_activations.columns:
      mean_inc = incidence_scenarios.mean('scenario').sel(
          time=slice(date + obs_delay, None)).mean('time')
      # Already activated sites can't be activated further, so zero out their
      # incidence to avoid picking them.
      mean_inc.loc[dict(location=already_activated)] = 0.0
      for group, group_idx in self.group_to_idx.items():
        num_to_activate = self.allowed_activations[date][group]
        x = mean_inc.isel(location=group_idx)
        x = x.sortby(x, ascending=False)
        locations = x.location.values[:num_to_activate]
        slice_to_activate = dict(location=locations, time=slice(date, None))
        site_activation.loc[slice_to_activate] = 1.0
        already_activated.extend(locations)
    return site_activation


def optimize_params(c,
                    incidence_scenarios,
                    parameterizer,
                    loss_fn=None,
                    optimization_params=None,
                    verbose=True):
  """Modifies c in place subject to parameterizer to minimize a loss function.

  Picks params and runs parameterizer.xr_apply_params(c, params) to minimize
  loss_fn on the resulting trial given incidence_scenarios. The final optimized
  params are stored in c['final_params'].

  Args:
    c: xr.Dataset specifying the trial with all data_vars required to call
      sim.recruitment and sim.control_arm_events.
    incidence_scenarios: xr.DataArray of shape [scenario, location, time], the
      forecast incidence.
    parameterizer: a Parameterizer specifying what the trial planner can
      control.
    loss_fn: optional function which takes a jax-ified trial object and returns
      a jnp.array of losses of shape [scenario]. Defaults to
      negative_mean_successiness.
    optimization_params: optional dict of stuff related to how to do the
      optimization.
    verbose: if True, print some stuff.
  """
  # Run non-differentiable simulation once, because it does more error checking.
  participants = sim.recruitment(c)
  sim.control_arm_events(c, participants, incidence_scenarios)

  c_ = sim.JaxDataset(c)
  incidence_scenarios_ = jnp.asarray(
      incidence_scenarios.transpose('scenario', 'location', 'time'))
  _, recruitment_fn, _ = sim.RECRUITMENT_REGISTRY[sim.get_recruitment_type(c)]
  historical_events = c.historical_control_arm_events
  if 'location' in historical_events.dims:
    historical_events = c.historical_control_arm_events.sum('location')
  historical_events_ = jnp.array(historical_events.values)

  if loss_fn is None:
    loss_fn = negative_mean_successiness

  def loss(params):
    parameterizer.apply_params(c_, params)
    c_.participants = recruitment_fn(c_)
    control_arm_events_ = sim.differentiable_control_arm_events(
        c_, c_.participants, incidence_scenarios_)
    historical_ = jnp.broadcast_to(
        historical_events_,
        control_arm_events_.shape[:1] + historical_events_.shape)
    c_.control_arm_events = jnp.concatenate([historical_, control_arm_events_],
                                          axis=1)
    return loss_fn(c_).mean()

  if optimization_params is None:
    optimization_params = dict()
  min_steps = optimization_params.get('min_steps', 40)
  max_steps = optimization_params.get('max_steps', 200)
  epsilon = optimization_params.get('epsilon', 1e-3)
  smoothing_window = optimization_params.get('smoothing_window', 20)
  learning_rate = optimization_params.get('learning_rate', 0.1)
  optimizer = optimization_params.get('optimizer', optim.Adam(learning_rate))
  initial_params = parameterizer.init_params()
  optimizer = optimizer.create(jnp.array(initial_params.values))

  loss_curve = []
  while True:
    loss_value, grad = jax.value_and_grad(loss)(optimizer.target)
    loss_curve.append(float(loss_value))
    optimizer = optimizer.apply_gradient(grad)
    step = len(loss_curve)
    if (step > min_steps and
        (loss_curve[-smoothing_window] - loss_curve[-1]) < epsilon):
      # Not much progress recently. Call it a day.
      break
    if step >= max_steps:
      print('Hit max step limit. You can control this exit condition by '
            'setting max_steps in optimized_params.')
      break
    if verbose and (step % 10 == 0):
      print(f'step {step}, loss value {loss_curve[-1]}')

  final_params = np.array(optimizer.target)
  c['final_params'] = xr.DataArray(final_params, coords=initial_params.coords)
  parameterizer.xr_apply_params(c, c.final_params)


def optimize_dynamic_activation(c,
                                incidence_scenarios,
                                loss_fn=None,
                                optimization_params=None):
  # Just here for backwards compatibility. Prefer calling optimize_params
  # directly.
  parameterizer = DynamicSiteActivation(c)
  optimize_params(c, incidence_scenarios, parameterizer, loss_fn,
                  optimization_params)
  c['site_activation'] = c.site_activation.round()


def optimize_static_activation(c,
                               incidence_scenarios,
                               loss_fn=None,
                               optimization_params=None):
  # Just here for backwards compatibility. Prefer calling optimize_params
  # directly.
  parameterizer = StaticSiteActivation(c)
  optimize_params(c, incidence_scenarios, parameterizer, loss_fn,
                  optimization_params)
  c['site_activation'] = c.site_activation.round()


def negative_mean_successiness(c):
  """Negative mean successiness over the course of the trial."""
  if c.needed_control_arm_events.size > 1:
    center = float(c.needed_control_arm_events.mean())
    width = float(c.needed_control_arm_events.std())
  else:
    center = float(c.needed_control_arm_events)
    width = center / 3

  cum_events = c.control_arm_events.cumsum(axis=-1)
  successiness = nn.sigmoid((cum_events - center) / width)
  return -successiness.mean(axis=-1)
