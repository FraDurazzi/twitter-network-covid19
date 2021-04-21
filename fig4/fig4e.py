import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('..')
from helpers.analysis_helpers import save_fig, plot
import matplotlib.dates as mdates
import matplotlib
import glob
import numpy as np

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
matplotlib.rcParams.update(new_rc_params)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

DATA_DIR = os.path.join('..', 'data', 'fig4')

community_to_types = {
        'A': 'Other',
        'B': 'International expert',
        'C': 'Political',
        'D': 'National expert',
        'E': 'Other',
        'F': 'Political',
        'G': 'International expert',
        'H': 'Political',
        'I': 'National expert',
        'J': 'National expert',
        'K': 'Political',
        'L': 'Political',
        'M': 'National expert',
        'N': 'Other',
        'O': 'Other'}

def read_plot_data(overwrite=False):
    f_path_cached = os.path.join(DATA_DIR, 'h_index_vs_time_merged.parquet')
    df = pd.read_parquet(f_path_cached)
    return df

def read_network_data():
    logger.info('reading community data...')
    comm_data = os.path.join(DATA_DIR, 'com_of_user_letters_1_30.pickle')
    with open(comm_data, 'rb') as f:
        df = pickle.load(f)
    df = pd.Series(df, name='community')
    df.index.name = 'user.id'
    df = df.reset_index()
    return df


@plot
def main():
    logger.info('Reading plot data...')
    df = read_plot_data(overwrite=False)
    df = df.reset_index()
    # rename
    df['super_community'] = df.super_community.apply(lambda s: s if s != 'National expert' else 'National elite')
    logger.info('Computing ranks...')
    for centroid_day, grp in df.groupby('centroid_day'):
        df.loc[grp.index, 'h_index'] = grp.h_index.rank(method='first', ascending=False)
        df.loc[grp.index, 'num_retweets'] = grp.num_retweets.rank(method='first', ascending=False)
    df = df.groupby(['centroid_day', 'super_community']).mean().reset_index()
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {'International expert': palette[4], 'National elite': palette[1], 'Political': palette[2], 'Other': '.5'}

    fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)
    axes=axes.flatten()

    min_date = df.centroid_day.min()
    for ax, col in zip(axes, color_dict.keys()):
        _df = df[df.super_community == col]
        X = []
        Y = []
        U = []
        V = []
        c = []
        start = []
        end = []
        for centroid_day, grp in _df.groupby('centroid_day'):
            if centroid_day == min_date:
                last_h_index = grp.h_index.mean()
                last_retweets = grp.num_retweets.mean()
                start = [last_h_index, last_retweets]
                continue
            h_index = grp.h_index.mean()
            num_retweets = grp.num_retweets.mean()
            X.append(last_h_index)
            Y.append(last_retweets)
            U.append(h_index - last_h_index)
            V.append(num_retweets - last_retweets)
            last_h_index = h_index
            last_retweets = num_retweets
            c.append(color_dict[col])
        end = [last_h_index, last_retweets]

        _min = 800
        _max = 3200
        ax.plot([_min, _max], [_min, _max], ls='dashed', lw=.5, c='.15')
        ax.plot(*start, c=color_dict[col], marker='s', ms=3)
        ax.plot(*end, c=color_dict[col], marker='o', ms=3)
        # for x, y, u, v, color, alpha in zip(X, Y, U, V, c, np.linspace(.1, 1, len(c))):
        #     ax.quiver(x, y, u, v, scale_units='xy', angles='xy', scale=1, color=color, alpha=alpha)
        ax.quiver(X, Y, U, V, scale_units='xy', angles='xy', scale=1, color=c, width=.01)
        ax.set_title(col, fontsize=7)
        ax.grid(True)
        ax.set_ylim((_max, _min))
        ax.set_xlim((_max, _min))
        ax.set_aspect('equal')
        plt.locator_params(axis='y', nbins=3)

    #axes[0].set_ylabel('Avg. rank retweets $r_{rt}$')
    fig.text(0.04, 0.51, r'Avg. rank retweets $r_{rt}$', va='center', rotation='vertical', fontsize=7)
    fig.text(0.5, 0.15, r'Avg. rank $h$-index $r_{h}$', ha='center', va='top', fontsize=7)
    fig.subplots_adjust(wspace=.07)

    save_fig(fig, 'fig4e', 1, dpi=600, plot_formats=['png', 'pdf','svg'])


if __name__ == "__main__":
    main()
