import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import DelegatesUtils
OUTPUTS_DIR = "./outputs/"

backend = DelegatesUtils.Backend.pgf
operation_mode = DelegatesUtils.OperationMode.convolution
[save_path, graph_params] = DelegatesUtils.pyplot_backend_change(backend, operation_mode)
(_layer, _flips, _bit, _acc_deg, _loss) = DelegatesUtils.mask_fields()

# Constants and data loading
DATA_PATH = OUTPUTS_DIR + 'delegate_16-50-58_convolution_2024-01-27.csv'
df = pd.read_csv(DATA_PATH)

""" Best colors:
    Points          Limits          Line
    violet          darkmagenta     purple
    lightcoral      darkred         maroon
    #82BFED         #004F8C         #004880"""
colors = {"points": "#82BFED", "line": "#004880", "boundaries": "#004F8C"}
LAYERS = ("conv2d/", "conv2d_1/", "conv2d_2/", "last/")
for layer in LAYERS:
    save_name = 'delegate_conv_acc_degradation_' + layer[:-1] + graph_params.extension

    flips = np.unique(df[_flips])
    flips = flips[~np.isnan(flips)].astype(int)
    bits_affected = np.unique(df[_bit])
    bits_affected = bits_affected[~np.isnan(bits_affected)].astype(int)

    # Graphs definition
    graph_params.update_number_graphs(n_graphs = len(flips), n_base = 3)

    fig, ax = plt.subplots(graph_params.n_rows, graph_params.n_cols, figsize = (graph_params.fig_width, graph_params.fig_height), sharey = True)
    # Transparent background
    fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))

    if backend == DelegatesUtils.Backend.normal:
        fig.suptitle(f"Layer: {layer[:-1]}", fontsize = graph_params.fontsize * 1.25)
    # For tight_layout h_pad and w_pad only work when the labels and ticks between graphs are not hidden
    # Which here it is the case
    fig.tight_layout(rect = graph_params.rect_window)
    
    decimals = [1, 1, 1, 1]
    # yaxis_major_locators = [.2, .4, .5, .8]
    # yaxis_minor_locators = [4, 4, 4, 2]

    for i, flip in enumerate(flips):
        graph_pos = DelegatesUtils.convert_position(i, n_rows = graph_params.n_rows, n_cols = graph_params.n_cols)
        axes = ax[graph_pos]
        axes.yaxis.set_major_formatter(ticker.PercentFormatter(decimals = decimals[i]))
        condition = (df[_layer].notna()) & (df[_flips] == flip) & (df[_layer] == layer)
        axes.plot(df.loc[condition][_bit], 
                  pd.to_numeric(df.loc[condition][_acc_deg])*100, 
                  color = colors['points'], 
                  linewidth = 0, 
                  marker = 's', 
                  alpha = 0.05, 
                  markersize = graph_params.markersize, 
                  markeredgewidth = 0)
        # axes.set_title('Multiplications affected: ' + str(flip))
        wrapped_title = DelegatesUtils.text_wrapper(fig, axes, 
                                                    label = 'Multiplications affected: ' + str(flip), 
                                                    fontsize = graph_params.fontsize)
        axes.text(x = 0.5, y = 0.03, 
                  s = wrapped_title, 
                  horizontalalignment = "center", 
                  verticalalignment = "bottom", 
                  transform = axes.transAxes,
                  fontsize = graph_params.fontsize)
        x_label = "Accumulator bit position disrupted"
        y_label = "Accuracy degradation"
        graph_params.configure_graphs(fig, axes, graph_pos, y_label, x_label)
        axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
        axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        # axes.yaxis.set_major_locator(ticker.MultipleLocator(yaxis_major_locators[i]))
        # axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(yaxis_minor_locators[i])) 
        axes.yaxis.set_minor_locator(ticker.AutoMinorLocator()) 
        axes.set_xlim([np.min(bits_affected) - .25, np.max(bits_affected) + .25])

        # Trend lines
        averages = []
        stds = []
        bits_array = np.unique(df.loc[condition][_bit])
        for j in bits_array:
            new_condition = condition & (df[_bit] == j)
            averages.append(pd.to_numeric(df.loc[new_condition][_acc_deg]).mean())
            stds.append(pd.to_numeric(df.loc[new_condition][_acc_deg]).std())   
        averages = np.array(averages)
        axes.plot(bits_array, averages*100, alpha = 1.00, color = colors['line'], linewidth = graph_params.trend_linewidth)
        axes.plot(bits_array, (averages + stds)*100, '--', alpha = 0.95, color = colors['boundaries'], linewidth = 0.6*graph_params.trend_linewidth)
        axes.plot(bits_array, (averages - stds)*100, '--', alpha = 0.95, color = colors['boundaries'], linewidth = 0.6*graph_params.trend_linewidth)

    # Controls the reserved space between the graphs
    plt.subplots_adjust(wspace = graph_params.wspace, hspace = graph_params.hspace)
    save_name = 'delegate_conv_acc_degradation_' + layer[:-1] + ".pdf"
    plt.savefig(save_path + save_name, bbox_inches = 'tight', format = "pdf")
plt.show()
plt.close()