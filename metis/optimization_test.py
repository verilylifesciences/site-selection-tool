# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Tests for metis.optimization."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import xarray as xr
from metis import optimization
from metis import sim_test_util
from metis import util


def make_pivot_activation_inputs():
  """Returns a c, site_to_group mapping, and allowed_activations table."""
  # Make 3 groups with 5 sites each.
  groups = ['A', 'B', 'C']
  sites = [f'{group}{i}' for group in groups for i in range(5)]
  site_to_group = pd.Series([x[:1] for x in sites], index=sites, name='group')

  time = pd.date_range('2020-10-01', '2020-10-10')
  site_activation = xr.DataArray(
      np.zeros((len(sites), len(time))),
      dims=('location', 'time'),
      coords=(sites, time))
  c = xr.Dataset({'site_activation': site_activation})

  decision_days = time[[1, 3, 5]]
  allowed_activations = pd.DataFrame(
      np.array([
          [1, 2, 2],  # group A
          [0, 1, 2],  # group B
          [0, 0, 1],  # group C
      ]),
      index=pd.Index(groups, name='group'),
      columns=pd.Index(decision_days, name='time')).astype('float32')

  return c, site_to_group, allowed_activations


def new_site_activation(site_activation: xr.DataArray) -> xr.DataArray:
  new_activation = site_activation - site_activation.shift(time=1, fill_value=0)
  new_activation = xr.where(new_activation < 0.0, 0.0, new_activation)
  return new_activation


class ParameterizerTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(force_hit_cap=False, can_deactivate=True),
      dict(force_hit_cap=False, can_deactivate=False),
      dict(force_hit_cap=True, can_deactivate=False),
      # dict(force_hit_cap=True, can_deactivate=True),  # not supported
  )
  def test_pivot_constraints_satisfied(self, force_hit_cap, can_deactivate):
    c, site_to_group, allowed_activations = make_pivot_activation_inputs()

    parameterizer = optimization.PivotTableActivation(c, site_to_group,
                                                      allowed_activations,
                                                      force_hit_cap,
                                                      can_deactivate)
    np.random.seed(0)
    params = parameterizer.init_params()
    parameterizer.xr_apply_params(c, params)

    # Activations must be between 0 and 1.
    self.assertTrue(
        np.logical_and(0 <= c.site_activation, c.site_activation <= 1.0).all())

    new_activation = new_site_activation(c.site_activation)

    # If deactivation is not allowed, check that none has occurred.
    if not can_deactivate:
      expected_activation = new_activation.cumsum('time')
      xr.testing.assert_allclose(expected_activation, c.site_activation)

    # No new activations on dates other than decision days.
    decision_days = list(allowed_activations.columns)
    non_decision_days = [
        x for x in new_activation.time.values if x not in decision_days
    ]
    actual = new_activation.sel(time=non_decision_days)
    expected = xr.zeros_like(actual)
    xr.testing.assert_equal(expected, actual)

    # Check new activations by group is bounded as should be.
    new_activation_by_group = new_activation.sel(
        time=decision_days).to_pandas().groupby(site_to_group).sum()
    # Since apply_params uses float32 arithmetic, we may slightly exceed the
    # allowed limit when force_hit_cap is true.
    epsilon = 1e-6
    self.assertTrue(
        (new_activation_by_group <= allowed_activations + epsilon).all().all())

    # If we're required to hit the cap, check that we do.
    if force_hit_cap:
      pd.testing.assert_frame_equal(new_activation_by_group,
                                    allowed_activations)


def c_and_scenarios_to_test_optimizer(rand=None):
  if rand is None:
    rand = np.random.RandomState(0)
  c = sim_test_util.c_to_test_recruitment(rand)
  c = c.merge(sim_test_util.c_to_test_events(rand))
  c['historical_control_arm_events'] = util.sum_all_but_dims(
      ['location', 'historical_time'], xr.zeros_like(c.historical_participants))
  needed_events = xr.DataArray([20, 15, 10],
                               dims=('analysis',),
                               coords=(['one', 'two', 'three'],))
  c['needed_control_arm_events'] = needed_events
  _, incidence_scenarios = sim_test_util.participants_and_forecast()
  return c, incidence_scenarios


class OptimizationTest(absltest.TestCase):

  def test_optimizer_runs_static(self):
    c, incidence_scenarios = c_and_scenarios_to_test_optimizer()
    optimization.optimize_static_activation(c, incidence_scenarios)

  def test_optimizer_runs_dynamic(self):
    c, incidence_scenarios = c_and_scenarios_to_test_optimizer()
    optimization.optimize_dynamic_activation(c, incidence_scenarios)

  def test_optimizer_runs_pivot(self):
    c, incidence_scenarios = c_and_scenarios_to_test_optimizer()

    # Say you can activate one site each week for the first three weeks. All
    # sites are in one group.
    sites = c.location.values
    site_to_group = pd.Series(
        ['the_group'] * len(sites), index=sites, name='group')
    decision_days = c.time.values[[0, 7, 14]]
    allowed_activations = pd.DataFrame(
        np.array([[1, 1, 1]]),
        index=pd.Index(['the_group'], name='group'),
        columns=pd.Index(decision_days, name='time')).astype('float32')
    parameterizer = optimization.PivotTableActivation(c, site_to_group,
                                                      allowed_activations)

    optimization.optimize_params(c, incidence_scenarios, parameterizer)

  def test_zero_risk_error(self):
    c, incidence_scenarios = c_and_scenarios_to_test_optimizer()
    c['incidence_scaler'] = xr.zeros_like(c.incidence_scaler)
    with self.assertRaisesRegex(ValueError,
                                'impossible to account for incidence!'):
      optimization.optimize_static_activation(c, incidence_scenarios)


if __name__ == '__main__':
  absltest.main()
