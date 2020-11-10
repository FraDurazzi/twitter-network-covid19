import pandas as pd
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
import joblib
sys.path.append('..')
from helpers.analysis_helpers import save_fig, plot
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

DATA_DIR = os.path.join('..', 'data', 'fig4')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


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

@plot
def main():
    df = read_plot_data(overwrite=False)
    df = df.reset_index()
    for centroid_day, grp in df.groupby('centroid_day'):
        df.loc[grp.index, 'h_index'] = grp.h_index.rank(method='first', ascending=False)
        df.loc[grp.index, 'num_retweets'] = grp.num_retweets.rank(method='first', ascending=False)
    df = df.groupby(['centroid_day', 'super_community']).mean().reset_index()
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {'International expert': palette[4], 'National expert': palette[1], 'Political': palette[2], 'Other': '.5'}

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.4))

    lw = .8
    ms = 4
    for sc, grp in df.groupby('super_community'):
        axes[0].plot(grp.centroid_day.values, grp.h_index, c=color_dict[sc], label=sc, ms=ms, lw=lw, marker='.')
        axes[1].plot(grp.centroid_day.values, grp.num_retweets, c=color_dict[sc], label=sc, ms=ms, lw=lw, marker='.')

    _min = 600
    _max = 3200
    axes[0].set_ylim((_max, _min))
    axes[1].set_ylim((_max, _min))

    axes[0].set_ylabel('Avg. rank h-index $r_h$')
    axes[1].set_ylabel('Avg. rank retweets $r_{rt}$')

    for ax in axes:
        ax.grid()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='x', direction='out', which='minor', size=2)
        ax.tick_params(axis='x', direction='out', which='major', size=2)
        ax.tick_params(axis='y', which='major', direction='out', size=2)
        ax.locator_params(nbins=4)

    sns.despine()
    fig.tight_layout()

    save_fig(fig, 'fig4cd', 1, dpi=600, plot_formats=['png', 'pdf'])

if __name__ == "__main__":
    main()
