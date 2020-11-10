import os
import logging
import matplotlib.pyplot as plt
from functools import wraps

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True

def get_plot_folder():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '..', 'plots'))

def save_fig(fig, fig_type, name, plot_formats=['png'], dpi=300):
    folder = os.path.join(get_plot_folder(), fig_type)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.abspath(os.path.join(folder, f_name))
    for fmt in plot_formats:
        f_path = f_name(fmt)
        logger.info(f'Writing figure file {f_path}')
        fig.savefig(f_path, bbox_inches='tight', dpi=dpi)

def plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        style_sheet = os.path.join(current_dir, '..', 'stylesheets', 'figure_small.mplstyle')
        plt.style.use(style_sheet)
        return func(*args, **kwargs)
    return wrapper
