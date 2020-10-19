# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# Sample Dockerfile for a custom Terra Docker image.
# For more detail, see https://support.terra.bio/hc/en-us/articles/360024737591-Make-a-Docker-container-image-the-easy-way-using-a-base-image

FROM us.gcr.io/broad-dsp-gcr-public/terra-jupyter-gatk:1.0.6
# https://github.com/DataBiosphere/terra-docker/blob/master/terra-jupyter-gatk/CHANGELOG.md

# Install ssh-agent and ssh-add for use with GitHub ssh keys.
USER root
RUN apt-get update && apt-get install -y keychain

# Install the Python package for Metis. Enable easy local edits by
# placing it in the home directory with write access for jupyter-user.
USER $USER
RUN mkdir -p $HOME/metis_pkg
COPY --chown=jupyter-user:users py/requirements.txt $HOME/metis_pkg/requirements.txt
COPY --chown=jupyter-user:users py/metis $HOME/metis_pkg/metis
ENV PYTHONPATH $PYTHONPATH:$HOME/metis_pkg

RUN pip3 install --user -r $HOME/metis_pkg/requirements.txt \
 && pip3 install --user pre-commit nbdime nbstripout
