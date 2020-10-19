# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Functions to read and write data from different file systems."""

import tensorflow as tf
import xarray as xr

open_fn = tf.io.gfile.GFile


def save_data(file_path, file_write_fn, file_open_fn, mode):
  """Write data to a file.

  Args:
    file_path: A string representing the path to the file to open.
    file_write_fn: A function to write the data with.
    file_open_fn: A function to open the file with.
    mode: A string representing the mode to use to open the file.
  """
  with file_open_fn(file_path, mode) as f:
    file_write_fn(f)


def load_data(file_path, file_read_fn, file_open_fn, mode):
  """Read a file into memory.

  Args:
    file_path: A string representing the path to the file to open.
    file_read_fn: A function to read the data with.
    file_open_fn: A function to open the file with.
    mode: A string representing the mode to use to open the file.

  Returns:
    The return value of file_read_fn.
  """
  with file_open_fn(file_path, mode) as f:
    data = file_read_fn(f.read())
  return data


def write_ville_to_netcdf(c, file_path, file_open_fn=None):
  """Writes a ville to disk.

  Args:
    c: xr.Dataset specifying the trial, suitable for running recruitment and
      event simulation (see sim.py for details about what data vars are
      expected).
    file_path: The path to write to.
    file_open_fn: A function to open the file with.
  """
  if file_open_fn is None:
    file_open_fn = open_fn

  # Have to make historical_time of length at least 1 because 0 dimension size
  # is used as a sentinel value in netcdf:
  # https://github.com/scipy/scipy/blob/1eae2ea615d9298e938a335ff2bc86ce345cd247/scipy/io/netcdf.py#L434
  def extend(historical, future):
    return xr.concat(
        [historical,
         future.isel(time=0).rename(time='historical_time')], 'historical_time')

  historical_participants = extend(c.historical_participants, c.participants)
  historical_control_arm_events = extend(
      c.historical_control_arm_events,
      c.control_arm_events.isel(scenario=0, drop=True))
  historical_site_activation = extend(c.historical_site_activation,
                                      c.site_activation)
  historical_incidence = extend(
      c.historical_incidence,
      c.incidence_model.isel(model=0, sample=0, drop=True))
  c = c.drop_vars([
      'historical_participants', 'historical_control_arm_events',
      'historical_site_activation', 'historical_incidence', 'historical_time'
  ])
  c['historical_participants'] = historical_participants
  c['historical_control_arm_events'] = historical_control_arm_events
  c['historical_site_activation'] = historical_site_activation
  c['historical_incidence'] = historical_incidence

  file_write_fn = lambda f: f.write(c.to_netcdf())
  save_data(file_path, file_write_fn, file_open_fn, 'wb')


def load_ville_from_netcdf(file_path, file_open_fn=None):
  """Read a ville into memory.

  Args:
    file_path: A string representing the path to the ville to open. We assume
      the ville is stored as a netCDF file.
    file_open_fn: A function to open the file with.

  Returns:
    An xr.Dataset representing the trial.
  """
  if file_open_fn is None:
    file_open_fn = open_fn
  c = load_data(file_path, xr.load_dataset, file_open_fn, 'rb')
  # Undo the historical_time hack used in write_ville_to_netcdf.
  return c.isel(historical_time=slice(None, -1))
