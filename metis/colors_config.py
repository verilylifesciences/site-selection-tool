# Copyright 2020 Verily Life Sciences LLC
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

import matplotlib as mpl
import matplotlib.pyplot as plt

"""Library level constants for plotting colors."""

# Settings used to color bar charts
BAR_CHART_CMAP = plt.cm.PiYG
BAR_CHART_NORM = mpl.colors.Normalize(vmin=-100, vmax=100)
DIFF_NORM = mpl.colors.Normalize(vmin=-10, vmax=10)
BAR_HEIGHT = 0.6
BAR_CHART_FACECOLOR = '#d6d6d6'

TRANSPARENT_HIST_COLOR = (1, 1, 1, 0)
LINE_WIDTH = 2.0

# Settings used to color vlines on bar charts
VLINE_HEIGHT = 1.2 * BAR_HEIGHT
VLINE_COLOR = '#000000'

# Dictionary of styles to use when plotting multiple villes
ms = 12

ville_styles = {
    'highlight_ville_1': {'color': '#ff00ff', 'markersize': ms},
    'highlight_ville_2': {'color': '#00d0db', 'markersize': ms},
    'gray_ville_1': {'color': '#b6b6b6', 'markersize': ms},
    'gray_ville_2': {'color': '#929292', 'markersize': ms},
    'gray_ville_3': {'color': '#6d6d6d', 'markersize': ms},
    'gray_ville_4': {'color': '#494949', 'markersize': ms},
    'gray_ville_5': {'color': '#242424', 'markersize': ms},
    'gray_ville_6': {'color': '#000000', 'markersize': ms},
}
