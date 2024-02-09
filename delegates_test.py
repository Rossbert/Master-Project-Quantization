import numpy as np
import matplotlib
import math
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
import DelegatesUtils
from enum import Enum
#from keras.engine.functional import Functional
from keras.src.engine.functional import Functional
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# plt.style.use('bmh')

def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig


OUTPUTS_DIR = "./outputs/"

# xkcd_fig = plot_colortable(mcolors.XKCD_COLORS)
# xkcd_fig.savefig("XKCD_Colors.png")
# plt.show()

# Accuracy degradation histogram model: 'delegate_16-50-58_convolution_2024-01-27.csv' layer: "conv2d/"
LAYER = "conv2d/"
backend = DelegatesUtils.Backend.normal
[save_path, base_fig_size, extension] = DelegatesUtils.pyplot_backend_change(backend)

# Constants and data loading
DATA_PATH = OUTPUTS_DIR + 'delegate_16-50-58_convolution_2024-01-27.csv'
save_name = 'delegate_hist_conv_acc_degradation_' + LAYER[:-1] + extension
df = pd.read_csv(DATA_PATH)

(_layer, _flips, _bit, _acc_deg, _loss) = DelegatesUtils.mask_fields()
flips = np.unique(df[_flips])
flips = flips[~np.isnan(flips)].astype(int)

bits = (13, 17, 25, 31)
factor = 0.5
n_rows = len(bits)
n_cols = len(flips)
fig, ax = plt.subplots(n_rows, n_cols, figsize = (base_fig_size * n_cols, base_fig_size * n_rows * factor))
if backend == DelegatesUtils.Backend.normal:
    rect = [0, 0, 1, 0.99]
    fig.suptitle(f"Histogram layer: {LAYER[:-1]}")
else:
    rect = [0, 0, 1, 1]
# Controls the *extra* padding of the graphs
fig.tight_layout(h_pad = 0.0, w_pad = 0.0, rect = rect)

# colors_list = ["#82BFED", "#5CAAE6", "#4784B3", "#004F8C", "#004880"] # blue palette
colors_list = ["#ED8682", "#E6605C", "#B34B47", "#8C0500", "#800400"] # red palette

list_xmax = []
list_xmin = []
list_ymax = []
for column, flip in enumerate(flips):
    condition = (df[_flips] == flip) & (df[_bit].isin(bits)) & (df[_layer] == LAYER)
    list_xmax.append(df.loc[condition][_acc_deg].max())
    list_xmin.append(df.loc[condition][_acc_deg].min())

delta = np.min(np.array(list_xmax) - np.array(list_xmin))
x_grid_step = float(f"{delta / 5:.{1}g}")*100

bin_grid_number = 40
for row, bit in enumerate(bits):
    for column, flip in enumerate(flips):
        graph_pos = (row, column)
        ax[graph_pos].xaxis.set_major_formatter(ticker.PercentFormatter(decimals = 1))
        condition = (df[_layer].notna()) & (df[_flips] == flip) & (df[_bit] == bit) & (df[_layer] == LAYER)
        data = pd.to_numeric(df.loc[condition][_acc_deg])*100
        binwidth = (list_xmax[column] - list_xmin[column])*100/bin_grid_number
        [values, bins, patches] = ax[graph_pos].hist(data, 
                                                     bins = np.arange(min(data), max(data) + binwidth, binwidth), 
                                                     color = colors_list[row],
                                                     alpha = 1.0,
                                                     density = True)
        list_ymax.append(values.max())
        ax[graph_pos].grid()
        ax[graph_pos].grid(which = 'major', linestyle = '-', alpha = 0.50, linewidth = 0.50)
        ax[graph_pos].grid(which = 'minor', linestyle = ':', alpha = 0.25, linewidth = 0.25)
        ax[graph_pos].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax[graph_pos].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax[graph_pos].set_xlim([list_xmin[column]*100 - binwidth, list_xmax[column]*100 + binwidth])
        if row == 0:
            ax[graph_pos].set_title('Multiplications affected ' + str(flip))
        if column != 0:
            ax[graph_pos].tick_params(axis = 'y', labelleft = False)
        if row < len(bits) - 1:
            ax[graph_pos].tick_params(axis = 'x', labelbottom = False)
        else:
            ax[graph_pos].set_xlabel('Accuracy degradation')
       
for row, bit in enumerate(bits):
    for column, flip in enumerate(flips):
        graph_pos = (row, column)
        ax[graph_pos].set_ylim([0, 1.025*np.max(list_ymax)])
        ax[graph_pos].legend(['Bit affected ' + str(bit)])

# Controls the reserved space between the graphs
plt.subplots_adjust(wspace = 0.02, hspace = 0.04)
# plt.savefig(save_path + save_name, bbox_inches = 'tight')
plt.show()
plt.close()