# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# Sample Dockerfile for a custom Terra Docker image.
# For more detail, see https://support.terra.bio/hc/en-us/articles/360024737591-Make-a-Docker-container-image-the-easy-way-using-a-base-image

FROM us.gcr.io/broad-dsp-gcr-public/terra-jupyter-gatk:1.0.8
# https://github.com/DataBiosphere/terra-docker/blob/master/terra-jupyter-gatk/CHANGELOG.md

RUN pip3 install --user git+https://github.com/verilylifesciences/site-selection-tool.git \
 && pip3 install --user pre-commit nbdime nbstripout
