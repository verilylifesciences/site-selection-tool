# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Tests for bsst.util."""

from absl.testing import absltest
import numpy as np
import scipy.stats
import xarray as xr
from bsst import util


class UtilTest(absltest.TestCase):

  def test_linear_interpolation_weights(self):
    rand = np.random.RandomState(seed=0)
    x = rand.rand(10)
    x.sort()
    xnew = np.linspace(0, 1, 20)
    weights = util.linear_interpolation_weights(x, xnew)
    np.testing.assert_allclose(np.dot(x, weights), xnew, atol=1e-7)
    for xnew_i, weights_i in zip(xnew, weights.T):
      lo, hi = np.nonzero(weights_i)[0]
      self.assertEqual(1, hi - lo)
      if lo > 0 and hi < len(x) - 1:
        self.assertTrue(x[lo] <= xnew_i <= x[hi])

  def test_linear_interpolation_weights_repeated_values(self):
    rand = np.random.RandomState(seed=0)
    x = rand.rand(3)
    x = np.concatenate([x, x])
    x.sort()
    xnew = np.linspace(0, 1, 5)
    with self.assertRaisesRegex(ValueError, 'cannot contain duplicates'):
        weights = util.linear_interpolation_weights(x, xnew)

  def test_quantile_conversion_weights(self):
    quantiles = np.array([0.025, 0.1, 0.25, .5, .75, .9, .975])
    new_quantiles = np.arange(1, 100) / 100
    samples = scipy.stats.norm.ppf(np.arange(1, 10000) / 10000)
    x = np.quantile(samples, quantiles)
    y_true = np.quantile(samples, new_quantiles)
    weights = util.quantile_conversion_weights(quantiles, new_quantiles)
    y_pred = np.dot(x, weights)
    np.testing.assert_allclose(y_pred, y_true, atol=0.002)

  def test_success_day(self):

    def day(n):
      """October n, 2020."""
      return np.datetime64(f'2020-10-{n:02}')

    time = np.arange(day(1), day(9), np.timedelta64(1, 'D'))
    cum_events = np.array([[1, 4, 9, 16, 25, 36, 49, 64],
                           [1, 2, 3, 4, 5, 6, 7, 8],
                           [2, 4, 8, 16, 32, 64, 128, 256]])
    cum_events = xr.DataArray(
        cum_events, dims=('scenario', 'time'), coords=([0, 1, 2], time))
    control_arm_events = cum_events - cum_events.shift(time=1, fill_value=0)

    # Check basic functionality when needed events is a singleton.
    needed_events = 3
    success_day = util.success_day(needed_events, control_arm_events)
    expected = np.array([day(2), day(3), day(2)])
    np.testing.assert_equal(success_day, expected)

    # Check basic functionality when needed events is a list.
    needed_events = [10, 50]
    success_day = util.success_day(needed_events, control_arm_events)
    expected = np.array([
        [day(4), np.datetime64('NaT'), day(4)],
        [day(8), np.datetime64('NaT'), day(6)],
    ])
    np.testing.assert_equal(success_day, expected.T)


if __name__ == '__main__':
  absltest.main()
