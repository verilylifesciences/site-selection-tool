# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Tests for metis.evaluation."""

from absl.testing import absltest
import numpy as np
import xarray as xr
from metis import evaluation


def day(n):
  """October n, 2020."""
  return np.datetime64(f'2020-10-{n:02}')


class EvaluationTest(absltest.TestCase):

  def test_success_day(self):
    time = np.arange(day(1), day(9), np.timedelta64(1, 'D'))
    cum_events = np.array([[1, 4, 9, 16, 25, 36, 49, 64],
                           [1, 2, 3, 4, 5, 6, 7, 8],
                           [2, 4, 8, 16, 32, 64, 128, 256]])
    cum_events = xr.DataArray(
        cum_events, dims=('scenario', 'time'), coords=([0, 1, 2], time))
    control_arm_events = cum_events - cum_events.shift(time=1, fill_value=0)

    # Check basic functionality when needed events is a singleton.
    needed_events = 3
    success_day = evaluation.success_day(needed_events, control_arm_events)
    expected = np.array([day(2), day(3), day(2)])
    np.testing.assert_equal(success_day, expected)

    # Check basic functionality when needed events is a list.
    needed_events = [10, 50]
    success_day = evaluation.success_day(needed_events, control_arm_events)
    expected = np.array([[day(4), day(9), day(4)], [day(8), day(9), day(6)]])
    np.testing.assert_equal(success_day, expected.T)


if __name__ == '__main__':
  absltest.main()
