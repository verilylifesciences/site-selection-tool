# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Utilities for fetching and munging public data and forecasts."""

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from metis import util

# By default, these will be populated from covid19-open-data. Overwrite these
# module variables if you want to use your own.
DEMOGRAPHICS = None
EPIDEMIOLOGY = None
INDEX = None


def demographics():
  global DEMOGRAPHICS
  if DEMOGRAPHICS is None:
    DEMOGRAPHICS = pd.read_csv(
        'https://storage.googleapis.com/covid19-open-data/v2/demographics.csv',
        index_col=0)
  return DEMOGRAPHICS


def index():
  global INDEX
  if INDEX is None:
    INDEX = pd.read_csv(
        'https://storage.googleapis.com/covid19-open-data/v2/index.csv',
        index_col=0)
  return INDEX


def epidemiology():
  global EPIDEMIOLOGY
  if EPIDEMIOLOGY is None:
    EPIDEMIOLOGY = pd.read_csv(
        'https://storage.googleapis.com/covid19-open-data/v2/epidemiology.csv')
    EPIDEMIOLOGY.rename(columns=dict(key='location', date='time'), inplace=True)
    EPIDEMIOLOGY['time'] = pd.to_datetime(EPIDEMIOLOGY['time'])
  return EPIDEMIOLOGY


def us_county_data():
  """An opencovid_key-indexed dataframe of county data."""
  keys = demographics().index.values.astype(str)
  us_keys = keys[np.char.startswith(keys, 'US_')]
  key_fips = []
  for key in us_keys:
    us_state_fips = key.split('_')
    if len(us_state_fips) == 3:
      key_fips.append((key, us_state_fips[2]))
  keys, fips = zip(*key_fips)

  us_counties = pd.Index(keys, name='location')
  county_df = pd.DataFrame(index=us_counties)
  county_df['fips'] = fips
  county_df['population'] = demographics().loc[us_counties, 'population']
  county_df['subregion1_name'] = index().loc[us_counties, 'subregion1_name']
  county_df['subregion2_name'] = index().loc[us_counties, 'subregion2_name']
  return county_df


def quantiles_to_samples(pred, num_samples):
  sample = np.linspace(0, 1, num_samples + 2)[1:-1]
  sample = xr.DataArray(
      sample, dims=('sample',), coords=(np.arange(len(sample)),))
  weights = util.quantile_conversion_weights(pred['quantile'].values,
                                             sample.values)
  weights = xr.DataArray(weights, coords=(pred['quantile'], sample['sample']))
  # Bound forecast incidence below by 0.
  return np.maximum(0.0, np.exp(np.log(pred + 1).dot(weights)) - 1)


def fetch_cdc_forecast(model, date_stamp, end_date=None, num_samples=None):
  """Returns forecast incidence of the given model from the given date.

  Forecasts are converted from the quantiles given to num_samples uniformly
  spaced quantiles. New case forecasts are converted to incidence forecasts
  using population data from opencovid. Forecasts are constant-extrapolated
  through end_date.

  Args:
    model: string, which model forecast to fetch (e.g. 'COVIDhub-ensemble')
    date_stamp: string, which date-stamped forecast to use (e.g. '2020-10-12')
    end_date: how far to constant-extrapolate. If omited, no extrapolation is
      done. Must be castable to np.datetime64.
    num_samples: number of uniformly-spaced quantiles to return in the sample
      dimension. If omitted, the original quantiles are returned with a
      "quantile" dimension instead of a "sample" dimension.

  Returns:
    An xr.DataArray of shape [sample, location, time], the forecast incidence
    from the model.
  """
  url = (f'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/'
         f'master/data-processed/{model}/{date_stamp}-{model}.csv')
  df = pd.read_csv(url)
  county_df = us_county_data()

  # Restrict to counties
  df = df[df.location.isin(county_df.fips)]

  fips_to_key = county_df.reset_index().set_index('fips')['location']
  df.loc[:, 'location'] = fips_to_key.loc[df.location].values

  # Restrict to the quantiles, discarding point predictions.
  df = df.query('type == "quantile"')

  df = df.rename(columns=dict(target_end_date='time'))
  df.loc[:, 'time'] = pd.to_datetime(df.time)
  df = df.set_index(['location', 'time', 'quantile'])['value']
  da = df.to_xarray()
  incidence = da / county_df.loc[da.location].population.to_xarray()
  if num_samples is not None:
    incidence = quantiles_to_samples(incidence, num_samples)
    incidence = incidence.transpose('sample', 'location', 'time')
  else:
    incidence = incidence.transpose('quantile', 'location', 'time')
  if end_date is not None:
    incidence = util.constant_extrapolate(incidence, np.datetime64(end_date))

  return incidence


def fetch_cdc_forecasts(model_dates, end_date='2021-04-01', num_samples=100):
  """A single array with incidence forecasts from multiple models."""
  forecasts_names = []
  for model, date_stamp in model_dates:
    forecast = fetch_cdc_forecast(model, date_stamp, end_date, num_samples)
    forecasts_names.append((forecast, model))
  return util.xr_stack('model', forecasts_names)


def fetch_opencovid_incidence():
  """Historical incidence for US counties from opencovid.

  Returns:
    An xr.DataArray of shape [location, time], the historical incidence.
  """
  county_df = us_county_data()

  # Construct ground truth incidence
  gt = epidemiology().set_index(['location',
                                 'time'])['new_confirmed'].to_xarray()
  us_counties = set(gt.location.values) & set(county_df.index.values)
  gt = gt.sel(location=list(us_counties))
  gt = gt / county_df.loc[gt.location].population.to_xarray()

  return gt


REF_TIME = np.datetime64('2020-11-01')
ONE_DAY = np.timedelta64(1, 'D')
ONE_SECOND = np.timedelta64(1, 's')


def to_float(datetime):
  return (datetime - REF_TIME) / ONE_DAY


def to_date(fl):
  return REF_TIME + fl * ONE_DAY


def assemble_forecast(full_gt, full_pred, site_df, time):
  """Uniformly spaced incidence forecast, indexed by site_id.

  Args:
    full_gt: an xr.DataArray of shape [location, time], the historical incidence
      for a given opencovid key.
    full_pred: an xr.DataArray of shape [model, sample, location, time], the
      forecast incidence for each opencovid key.
    site_df: a pd.DataFrame indexed by site ids with an "opencovid_key" column.
    time: a np.array of evenly-spaced np.datetime64's.

  Returns:
    An xr.DataArray of shape [model, sample, location, time], where the
    locations are site_ids from site_df and the time dimension matches the
    passed time argument. This is obtained by interpolating the cumulative
    incidence to the desired time points and reindexing by site_id instead of
    opencovid_key.
  """
  start_day, end_day = time[0], time[-1]
  time_resolution = time[1] - time[0]
  opencovid_keys = site_df.opencovid_key.unique()

  # Use ground truth for as many full time periods as you have it.
  sliced_pred = full_pred.sel(
      location=opencovid_keys,
      time=slice(full_gt.time.values[-1] + ONE_SECOND, None))

  # The first prediction is meant to cover the full time period prior to it, so
  # trim off any ground truth that cuts into that week.
  time_step = full_pred.time.values[1] - full_pred.time.values[0]
  sliced_gt = full_gt.sel(
      location=opencovid_keys,
      time=slice(None, sliced_pred.time.values[0] - time_step))

  cum_inc = xr.concat([sliced_gt, sliced_pred], 'time').cumsum('time')
  cum_inc = cum_inc.transpose('model', 'sample', 'location', 'time')

  interp_cum_incidence = scipy.interpolate.interp1d(
      to_float(cum_inc.time.values),
      cum_inc.values,
      kind='linear',
      axis=cum_inc.dims.index('time'))

  time = np.arange(
      start_day - time_resolution,  # extra will be cut off by diff
      end_day + time_resolution,  # extra to include endpoint
      time_resolution)
  interped_cum_inc = interp_cum_incidence(to_float(time))
  incidence = xr.DataArray(
      interped_cum_inc,
      dims=('model', 'sample', 'location', 'time'),
      coords=(cum_inc.model, cum_inc.sample, cum_inc.location,
              time)).diff('time')
  return util.reindex_by_site_id(site_df, incidence)
