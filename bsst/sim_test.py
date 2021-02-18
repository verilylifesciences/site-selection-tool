# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Tests for bsst.sim."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from bsst import sim
from bsst import sim_test_util


class SimTest(absltest.TestCase):

  def test_cumsum_capper(self):
    series = xr.DataArray([1, 0, 3, 4, 4, 4, 4, 4]).astype(float)
    capper = sim.cumsum_capper(series, 15)
    capper_ = sim.differentiable_cumsum_capper(series.values, 15, width=0.1)
    expected = xr.DataArray([1, 0, 3, 4, 4, 3, 0, 0]).astype(float)

    xr.testing.assert_equal(expected, capper * series)
    np.testing.assert_allclose(
        expected.values, capper_ * series.values, atol=1e-4)

  def test_cumsum_capper_nonnan_gradient(self):
    series_ = jnp.array([1, 0, 3, 4, 4, 4, 4, 4]).astype(float)
    fn = lambda x: sim.differentiable_cumsum_capper(x, 15, width=0.1).sum()
    grad = jax.grad(fn)(series_)
    self.assertFalse(jnp.isnan(grad).any())

  def test_recruitment_agreement(self):
    recruitment_fn, differentiable_fn, _ = sim.RECRUITMENT_REGISTRY['default']
    c = sim_test_util.c_to_test_recruitment()
    participants = recruitment_fn(c)
    participants_ = differentiable_fn(sim.JaxDataset(c), width=0.01)
    np.testing.assert_allclose(participants.values, participants_, rtol=1e-5)

  def test_events_agreement(self):
    participants, incidence_scenarios = (
        sim_test_util.participants_and_forecast())

    c = sim_test_util.c_to_test_events()
    events = sim.control_arm_events(c, participants, incidence_scenarios)
    events_ = sim.differentiable_control_arm_events(
        sim.JaxDataset(c), jnp.array(participants.values),
        jnp.array(incidence_scenarios.values))
    np.testing.assert_allclose(events.values, events_, atol=1e-6)

  def test_events_agreement_no_observation_delay(self):
    participants, incidence_scenarios = (
        sim_test_util.participants_and_forecast())

    c = sim_test_util.c_to_test_events()
    c['observation_delay'] = 0
    events = sim.control_arm_events(c, participants, incidence_scenarios)
    events_ = sim.differentiable_control_arm_events(
        sim.JaxDataset(c), jnp.array(participants.values),
        jnp.array(incidence_scenarios.values))
    np.testing.assert_allclose(events.values, events_, atol=1e-6)

  def test_bad_observation_delay_errors(self):
    participants, incidence_scenarios = (
        sim_test_util.participants_and_forecast())

    c = sim_test_util.c_to_test_events()

    c['observation_delay'] = -1
    with self.assertRaisesRegex(ValueError,
                                'Observation delay .* negative'):
      sim.control_arm_events(c, participants, incidence_scenarios)

    c['observation_delay'] = participants.time.size + 1
    with self.assertRaisesRegex(ValueError,
                                'Observation delay .* greater than the trial'):
      sim.control_arm_events(c, participants, incidence_scenarios)

  def test_nan_participants_error(self):
    participants, incidence_scenarios = (
        sim_test_util.participants_and_forecast())
    c = sim_test_util.c_to_test_events()
    participants[0, 0] = np.nan
    with self.assertRaisesRegex(ValueError, 'NaNs in participants array!'):
      sim.control_arm_events(c, participants, incidence_scenarios)

  def test_zero_risk_error(self):
    participants, incidence_scenarios = (
        sim_test_util.participants_and_forecast())
    c = sim_test_util.c_to_test_events()
    c['incidence_scaler'] = xr.zeros_like(c.incidence_scaler)
    with self.assertRaisesRegex(ValueError,
                                'impossible to account for incidence!'):
      sim.control_arm_events(c, participants, incidence_scenarios)


if __name__ == '__main__':
  absltest.main()
