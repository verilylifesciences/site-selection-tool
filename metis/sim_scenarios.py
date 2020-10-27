# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Scenario simulation code."""

import numpy as np
from scipy import stats
import scipy.special as special
import xarray as xr


def generate_scenarios_using_selection(incidence_flattened, selection):
  """Generates global incidence scenarios from selected incidence scenarios.

  Args:
    incidence_flattened: an xr.DataArray of shape [sample_flattened, location,
      time] of infection incidence.
    selection: a np.array of integers from 0 to
      incidence_flattened.sample_flattened.size - 1 and shape: (num_scenarios,
        incidence_flattened.location.size)

  Returns:
    An xr.DataArray of shape [scenario, location, time] representing infection
    incidence. Within each scenario, each location randomly selects one sample
    trajectory to follow.
  """
  num_options = incidence_flattened.sample_flattened.size
  num_scenarios = selection.shape[0]
  if selection.shape[1] != incidence_flattened.location.size:
    raise ValueError(f'location size mismatch: {selection.shape[1]}, '
                     f'{incidence_flattened.location.size}')
  selection = np.eye(num_options)[selection.ravel()].reshape(selection.shape +
                                                             (num_options,))
  selection = xr.DataArray(
      selection,
      dims=('scenario', 'location', 'sample_flattened'),
      coords=(np.arange(num_scenarios), incidence_flattened.location,
              incidence_flattened.sample_flattened))
  result = selection.dot(incidence_flattened, 'sample_flattened')
  return result.transpose('scenario', 'location', 'time')


def generate_scenarios_independently(incidence_flattened, num_scenarios=300):
  """Generates global incidence scenarios from local incidence scenarios.

  Assumes each local scenario is selected independently and uniformly.

  Args:
    incidence_flattened: an xr.DataArray of shape [sample_flattened, location,
      time] of infection incidence.
    num_scenarios: int, the desired number of global scenarios.

  Returns:
    An xr.DataArray of shape [scenario, location, time] representing infection
    incidence. Within each scenario, each location randomly selects one sample
    trajectory to follow.
  """
  num_options = incidence_flattened.sample_flattened.size
  selection = np.random.choice(
      np.arange(num_options),
      size=(num_scenarios, incidence_flattened.location.size))
  return generate_scenarios_using_selection(incidence_flattened, selection)


def sort_incidence_flattened(incidence_flattened):
  """Sorts samples within location by total incidence.

  The result is suitable for passing to
  generate_scenarios_with_latent_shift_structure.

  Args:
    incidence_flattened: xr.DataArray of shape [sample_flattened, location,
      time]

  Returns:
    xr.DataArray of shape [sample_flattened, location, time], the same data as
    incidence_flattened, but with the samples reordered so that
    result.sum('time') is sorted in the sample_flattened dimension.
  """
  # Sample index must be unique to stack/unstack.
  incidence_flattened = incidence_flattened.assign_coords(
      sample_flattened=np.arange(incidence_flattened.sample_flattened.size))

  sample_idx = incidence_flattened.sum('time').argsort(axis=0)
  location_idx = xr.ones_like(sample_idx) * np.arange(
      incidence_flattened.location.size)
  sample_idx = sample_idx.stack(
      sample_location=('sample_flattened', 'location'))
  location_idx = location_idx.stack(
      sample_location=('sample_flattened', 'location'))
  incidence_flattened = incidence_flattened.isel(
      location=location_idx, sample_flattened=sample_idx).unstack()
  return incidence_flattened


def get_incidence_flattened(pred, c):
  """Converts "raw" predictions for use in generate_scenarios_* methods.

  This method
  * rolls together model and sample dimensions into a single sample_flattened
    dimension.
  * restricts forecasts to the time period of the trial
  * sorts by cumulative incidence

  Args:
    pred: xr.DataArray of incidence forecasts, of shape [model, sample,
      location, time].
    c: xr.Dataset specifying the trial. This is used to determine what time to
      restrict to.

  Returns:
    An xr.DataArray of shape [sample_flattened, location, time] of incidence
    forecasts, where time is restricted to the time interval of the trial, and
    such that summing over the time dimension produces an array sorted in the
    sample dimension.
  """
  incidence_flattened = pred.stack(sample_flattened=('model', 'sample'))
  incidence_flattened = incidence_flattened.transpose('sample_flattened',
                                                      'location', 'time')
  included_days = np.logical_and(c.time.values[0] <= pred.time,
                                 pred.time <= c.time.values[-1])
  incidence_flattened = incidence_flattened.sel(time=included_days)
  incidence_flattened = incidence_flattened.assign_coords(
      sample_flattened=np.arange(incidence_flattened.sample.size))
  incidence_flattened = sort_incidence_flattened(incidence_flattened)
  return incidence_flattened.transpose('sample_flattened', 'location', 'time')


def generate_scenarios_with_latent_shift_structure(incidence_flattened,
                                                   strength_of_similarity,
                                                   rand,
                                                   num_scenarios=300):
  """Generates global incidence scenarios with a correlation in index selection.

  Args:
    incidence_flattened: an xr.DataArray of shape [sample_flattened, location,
      time] of infection incidence. The incidence forecasts should have a sorted
      order (e.g. low to high incidence) along the sample axis for the generated
      scenario to have the intended semantics.
     strength_of_similarity: (float) 0. <= . <= 1. The strength of "association"
       from location to location.
     rand: np.random.RandomState instance
     num_scenarios: int, the desired number of global scenarios.

  Returns:
    An xr.DataArray of shape [scenario, location, time] representing infection
    incidence. Within each scenario, each location randomly selects one sample
    trajectory to follow.
  """
  num_options = incidence_flattened.sample_flattened.size
  single_size = (incidence_flattened.location.size,)
  selection = repeated_choice_sim_with_latent_shift_structure(
      num_options, strength_of_similarity, single_size, num_scenarios, 0, rand)
  return generate_scenarios_using_selection(incidence_flattened, selection)


def normal_sim_with_latent_random_shift(correlation, size, rand):
  """Produce array of marginally N(0,1) vars with shared mean structure.

  The joint is the one generated by first choosing a random location
  from N(0, correlation) and then the observed being conditionally indep. normal
  with that location as the mean and variance = 1 - correlation, so that the
  marginal N(0,1) property is preserved.

  If correlation is 0, this is equivalent to:
    np.random.normal(size=size).
  If correlation is 1.0, this is equivalent to:
    np.random.normal() * np.ones_like(size).

  Arguments:
    correlation: (float) 0. <= . <= 1.
    size: The shape of the desired np.array.
    rand: np.random.RandomState instance

  Returns:
    rand_array
  """
  assert 0. <= correlation <= 1.
  m = rand.normal(scale=np.sqrt(correlation))
  std_of_resid = np.sqrt(1. - correlation)
  x = rand.normal(loc=m, scale=std_of_resid, size=size)
  return x


def uniform_sim_with_latent_shift_structure(strength_of_similarity, size, rand):
  x = normal_sim_with_latent_random_shift(strength_of_similarity, size, rand)
  return stats.norm.cdf(x)


def choice_sim_with_latent_shift_structure(a, strength_of_similarity, size,
                                           rand):
  u = uniform_sim_with_latent_shift_structure(strength_of_similarity, size,
                                              rand)
  if isinstance(a, int):
    return (u * a).astype(np.int32)
  else:
    raise ValueError('non-integer a not implemented')


def sim_repeatedly(simulator, num_repeats, rand, axis=-1):
  accum = []
  for _ in range(num_repeats):
    accum.append(simulator(rand))
  return np.stack(accum, axis=axis)


def repeated_choice_sim_with_latent_shift_structure(num_options,
                                                    strength_of_similarity,
                                                    single_size, num_repeats,
                                                    repeat_axis, rand):
  """Simulate an array of random choices."""

  def make_a_choice(rand):
    return choice_sim_with_latent_shift_structure(num_options,
                                                  strength_of_similarity,
                                                  single_size, rand)

  return sim_repeatedly(make_a_choice, num_repeats, rand, axis=repeat_axis)


def degrees_to_radians(degrees):
  return (2 * np.pi / 360.) * degrees


def great_circle_angle(lat1, lon1, lat2, lon2):
  ss = np.sin(lat1) * np.sin(lat2)
  cc = np.cos(lat1) * np.cos(lat2)
  return np.arccos(np.clip(ss + cc * np.cos(lon1 - lon2), 0., 1.))


def great_circle_distance_from_lat_lon(lat, lon):
  """We assume lat and lon are in degrees and xarrays with location dim."""
  lat1, lon1 = degrees_to_radians(lat), degrees_to_radians(lon)
  del lat, lon
  lat2, lon2 = (lat1.rename(location='location2'),
                lon1.rename(location='location2'))
  angles = great_circle_angle(lat1, lon1, lat2,
                              lon2).transpose('location', 'location2')
  earth_radius_km = 6371.
  dist_km = earth_radius_km * angles
  return dist_km


def matern_covariance(dist, variance, dist_scale, smoothness):
  """Compute Matérn covariance.

  https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

  Args:
    dist: an array of non-negative distances and shape (N, N).
    variance: (positive float) The variance at distance 0.
    dist_scale: (positive float) A distance scale (in the same units as dist).
    smoothness: (positive float) A Matern covariance with this smoothness is
      np.ceiling(smoothness) - 1. differentiable in the mean-square sense.

  Returns:
    A covariance matrix shaped like dist.
  """
  const = variance * 2**(1. - smoothness) / special.gamma(smoothness)
  x = (np.sqrt(2. * smoothness) * dist / dist_scale)
  cov = np.where(x == 0., variance,
                 const * x**smoothness * special.kv(smoothness, x))
  return cov


def mixed_matern_covariance(dist, global_variance, nugget_variance,
                            matern_variance, dist_scale, smoothness):
  matern_cov = matern_covariance(dist, matern_variance, dist_scale, smoothness)
  cov = (
      matern_cov + global_variance +
      nugget_variance * np.eye(matern_cov.shape[0]))
  return cov


def unit_marginal_mixed_matern_covariance(dist, global_logit, nugget_logit,
                                          dist_scale, smoothness):
  variances = special.softmax([global_logit, nugget_logit, 0.])
  global_variance, nugget_variance, matern_variance = variances
  cov = mixed_matern_covariance(dist, global_variance, nugget_variance,
                                matern_variance, dist_scale, smoothness)
  return cov


def shared_shift_covariance(rho, n):
  cov = np.full((n, n), rho)
  np.fill_diagonal(cov, 1.)
  return cov


def uniform_sim_with_latent_normal_covariance(cov, size, rand):
  """Convert multivariate normals to uniforms; assumes cov has 1's on diag."""
  x = rand.multivariate_normal(np.zeros((cov.shape[0],)), cov, size=size)
  return stats.norm.cdf(x)


def repeated_choice_sim_with_latent_normal_covariance(num_options, cov,
                                                      num_repeats, rand):
  r"""Generate dependent random choices based on a latent normal.

  Args:
    num_options: (positive int) choices run 0..(num_options-1)
    cov: a positive definite array of shape (N, N) with 1's on the diagonal.
    num_repeats: (positive integer) The number of rows of the return result,
      reflecting independent constructions.
    rand: An np.random.RandomState instance.

  Returns:
    int array with entries from 0 .. (num_options-1) and shape:
      (num_repeats, N)
  """
  u = uniform_sim_with_latent_normal_covariance(cov, num_repeats, rand)
  return (u * num_options).astype(np.int32)


def generate_scenarios_with_latent_normal_covariance(incidence_flattened,
                                                     cov,
                                                     rand,
                                                     num_scenarios=300):
  num_options = incidence_flattened.sample_flattened.size
  selection = repeated_choice_sim_with_latent_normal_covariance(
      num_options, cov, num_scenarios, rand)
  return generate_scenarios_using_selection(incidence_flattened, selection)


def generate_scenarios_with_mixed_matern_structure(incidence_flattened,
                                                   dist,
                                                   global_logit,
                                                   nugget_logit,
                                                   dist_scale,
                                                   smoothness,
                                                   rand,
                                                   num_scenarios=300):
  """Generates global incidence scenarios with a correlation in index selection.

  Args:
    incidence_flattened: an xr.DataArray of shape [sample_flattened, location,
      time] of infection incidence. The incidence forecasts should have a sorted
      order (e.g. low to high incidence) along the sample axis for the generated
      scenario to have the intended semantics.
    dist: an array of non-negative distances and shape (N, N).
    global_logit: (float) large positive values represent everything moving
      together. large negative values defer to the other variance components.
    nugget_logit: (float) large positive values represent everything moving
      independently. large negative values defer to the other variance
      components.
    dist_scale: (positive float) A distance scale (in the same units as dist).
    smoothness: (positive float) A Matern covariance with this smoothness is
      np.ceiling(smoothness) - 1. differentiable in the mean-square sense.
    rand: np.random.RandomState instance
    num_scenarios: int, the desired number of global scenarios.

  Returns:
    An xr.DataArray of shape [scenario, location, time] representing infection
    incidence. Within each scenario, each location randomly selects one sample
    trajectory to follow.
  """
  cov = unit_marginal_mixed_matern_covariance(dist, global_logit, nugget_logit,
                                              dist_scale, smoothness)
  return generate_scenarios_with_latent_normal_covariance(
      incidence_flattened, cov, rand, num_scenarios=num_scenarios)
