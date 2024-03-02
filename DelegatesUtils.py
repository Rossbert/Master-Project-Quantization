import matplotlib
import numpy as np
import textwrap
from enum import Enum, IntEnum
from typing import Tuple
from dataclasses import dataclass

def convert_position(x : int, n_rows : int = 1, n_cols : int = 1, axis : int = 0) -> int | Tuple[int, int]:
    """ Converts an integer position into a tuple that represents a n x 2 subplots"""
    if n_rows == 1:
        if x < n_cols:
            return x
        else:
            print("Error: position outside boundary")
            return None
    if n_cols == 1:
        if x < n_rows:
            return x
        else:
            print("Error: position outside boundary")
            return None
    if axis == 0:
        if x // n_cols < n_rows:
            return (x // n_cols, x % n_cols)
        else:
            print("Error: position outside boundary")
            return None
    else:
        if x // n_rows < n_cols:
            return (x % n_rows, x // n_rows)
        else:
            print("Error: position outside boundary")
            return None
        
class Backend(Enum):
    """ Enum that details the types of backends for matplotlib """
    normal = 1
    pgf = 2

class OperationMode(IntEnum):
    none = 0
    weights = 1
    convolution = 2

def get_extension(backend: Backend) -> str:
    """ Gets the extension of the backend """
    match backend:
        case Backend.normal:
            return ".png"
        case Backend.pgf:
            return ".pgf"
        case _ :
            return ""

def mask_fields() -> Tuple[str, str, str, str, str]:
    """ Gets the fields of the important variables """
    return ("layer_name", "n_bits_flipped", "bit_disrupted", "accuracy_degradation", "loss")

def text_wrapper(fig, axes, label: str, fontsize: int) -> str:
    """ Wraps the text """
    # Temporal text
    temp_text = axes.text(0, 0, label, fontsize = fontsize)
    text_box = temp_text.get_window_extent()
    temp_text.remove()
    # Figure values
    subplot_box = axes.get_position()
    fig_width, fig_height = fig.get_size_inches()
    # Length calculation
    text_width_inches = text_box.width/fig.get_dpi()
    subplot_width_inches = subplot_box.width * fig_width
    char_rate = len(label)/text_width_inches
    char_limit = np.floor(subplot_width_inches * char_rate).astype(int)
    wrapped_label = textwrap.fill(label, width = char_limit)
    return wrapped_label
@dataclass
class GraphParameters:
    _backend = Backend.normal
    def __init__(self, backend : Backend, operation_mode: OperationMode):
        self.backend = backend
        _backend = self.backend
        self.extension = get_extension(backend)
        if backend == Backend.normal:
            self.base_fig_size = 4.0
            self.graph_direction = 0
            self.rect_window = [0, 0, 1, 1.05]
            self.markersize = 6.75
            self.major_grid_alpha = 0.50
            self.major_grid_linewidth = 0.50
            self.minor_grid_alpha = 0.25
            self.minor_grid_linewidth = 0.25
            self.fontsize = 10.0
            self.box_linewidth = 0.5
            self.trend_linewidth = 1.5
            if operation_mode == OperationMode.weights:
                self.type_factor = 0.6
            else:
                self.type_factor = 1.0
        else:
            self.base_fig_size = 2.5
            self.graph_direction = 1
            self.rect_window = [0, 0, 1, 1]
            self.markersize = 2.5
            self.major_grid_alpha = 0.40
            self.major_grid_linewidth = 0.20
            self.minor_grid_alpha = 0.10
            self.minor_grid_linewidth = 0.10
            self.fontsize = 8.0
            self.box_linewidth = 0.2
            self.trend_linewidth = 0.9
            if operation_mode == OperationMode.weights:
                self.type_factor = 0.6
            else:
                self.type_factor = 1.0
        if operation_mode == OperationMode.weights:
            self.wspace = 0.06
            self.hspace = 0.06
        else:
            self.wspace = 0.02
            self.hspace = 0.02
    
    def update_number_graphs(self, n_graphs: int, n_base : int) -> None:
        """ Calculates the number of rows and columns according to the type of graph """
        if self.graph_direction:
            self.n_rows = n_base
            self.n_cols = np.ceil(n_graphs / n_base).astype(int)
        else:
            self.n_cols = n_base
            self.n_rows = np.ceil(n_graphs / n_base).astype(int)
        if self.n_cols == 1 and self.n_rows != 1:
            self.v_factor = 0.5
        else:
            self.v_factor = 1.0
        self.fig_width = self.base_fig_size * self.n_cols * self.type_factor
        self.fig_height = self.base_fig_size * self.n_rows * self.v_factor

    def configure_graphs(self, fig, axes, graph_pos: int | Tuple[int, int], y_label: str, x_label: str) -> None:
        """ Configures parameters for the graphs """
        axes.grid(which = 'major', linestyle = '-', alpha = self.major_grid_alpha, linewidth = self.major_grid_linewidth)
        axes.grid(which = 'minor', linestyle = ':', alpha = self.minor_grid_alpha, linewidth = self.minor_grid_linewidth)
        subplot_box = axes.get_position()
        size_limit = min(subplot_box.width, subplot_box.height)
        x_label_wrapped = text_wrapper(fig, axes, x_label, self.fontsize)
        axes.tick_params(axis = 'y', which = 'major', direction = 'in', width = self.box_linewidth, length = size_limit*12)
        axes.tick_params(axis = 'x', which = 'major', direction = 'in', width = self.box_linewidth, length = size_limit*12)
        axes.tick_params(axis = 'y', which = 'minor', direction = 'in', width = self.box_linewidth * 0.5, length = size_limit*8)
        axes.tick_params(axis = 'x', which = 'minor', direction = 'in', width = self.box_linewidth * 0.5, length = size_limit*8)
        # Adjusting the linewidth of the subplot box (spines)
        for spine in axes.spines.values():
            spine.set_linewidth(self.box_linewidth)
        if isinstance(graph_pos, tuple):
            row, column = graph_pos
        else:
            if self.n_cols == 1:
                column = 0
                row = graph_pos
            if self.n_rows == 1:
                row = 0
                column = graph_pos
        if row < self.n_rows - 1:
            axes.tick_params(axis = 'x', labelbottom = False)
        else:
            axes.set_xlabel(x_label_wrapped, fontsize = self.fontsize)
            axes.tick_params(axis = 'x', labelsize = self.fontsize)

        if column != 0:
            axes.tick_params(axis = 'y', labelbottom = False)
        else:
            # axes.set_ylabel(y_label, fontsize = self.fontsize)
            axes.tick_params(axis = 'y', labelsize = self.fontsize)

def pyplot_backend_change(backend : Backend, operation_mode: OperationMode) -> Tuple[str, GraphParameters]:
    """ Changes the backend to generate .png plots or .pgf for LATeX """
    graph_parameters = GraphParameters(backend, operation_mode)
    if backend == Backend.pgf:
        save_path = 'C:/Users/rosal/Documents/Files/Master Project/Paper/figures/'
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    else:
        save_path = './images/'
        matplotlib.use("module://matplotlib_inline.backend_inline")
        matplotlib.rcParams.update({
            "pgf.texsystem": "xelatex",
            'font.family': 'sans-serif',
            'text.usetex': False,
            'pgf.rcfonts': True,
        })
    return save_path, graph_parameters