# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""A setuptools based module for PIP installation of the Metis package."""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
# Get the requirements from the requirements file
requirements = (here / 'requirements.txt').read_text(encoding='utf-8')

setup(
  name='metis',
  version='0.0.1',
  license='BSD',

  description='Metis',

  python_requires='>=3.7',
  install_requires=requirements,
  packages=find_packages(),
  include_package_data=True,
  package_data = {
      '': ['demo_data/*'],
  },
)
