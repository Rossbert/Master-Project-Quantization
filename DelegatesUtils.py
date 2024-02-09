import matplotlib
from enum import Enum
from typing import Tuple

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

def pyplot_backend_change(extension : Backend) -> Tuple[str, int]:
    """ Changes the backend to generate .png plots or .pgf for LATeX """
    if extension == Backend.pgf:
        save_path = 'C:/Users/rosal/Documents/Files/Master Project/Report/'
        base_fig_size = 3
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    else:
        save_path = './images/'
        base_fig_size = 4.8
        matplotlib.use("module://matplotlib_inline.backend_inline")
        matplotlib.rcParams.update({
            "pgf.texsystem": "xelatex",
            'font.family': 'sans-serif',
            'text.usetex': False,
            'pgf.rcfonts': True,
        })
    return save_path, base_fig_size, get_extension(extension)