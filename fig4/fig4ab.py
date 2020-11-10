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
import numpy as np

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


def read_data(overwrite=False):
    f_path_cached = os.path.join(DATA_DIR, 'rank_rank_plot.parquet')
    df = pd.read_parquet(f_path_cached)
    df['h_index_rank'] = df.h_index.rank(method='first', ascending=False).astype(int)
    df['num_retweets_rank'] = df.num_retweets.rank(method='first', ascending=False).astype(int)
    return df

@plot
def fig4a():
    do_log = False
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {'International expert': palette[4], 'National expert': palette[1], 'Political': palette[2], 'Other': '.5'}
    df = read_data(overwrite=False)

    g = sns.jointplot(data=df, kind='scatter', x='h_index_rank', y='num_retweets_rank', alpha=.8, s=4, ec=None, hue='super_community', palette=color_dict.values(), hue_order=color_dict.keys(), height=2, marginal_ticks=False, space=.1)

    num_ranks = len(df) + 50
    if do_log:
        g.ax_joint.set_xlim((num_ranks, 1))
        g.ax_joint.set_ylim((num_ranks, 1))
    else:
        g.ax_joint.set_xlim((num_ranks, -30))
        g.ax_joint.set_ylim((num_ranks, -30))
    g.ax_joint.grid()
    g.ax_joint.get_legend().remove()
    g.ax_joint.set_xlabel(r'Rank $h$-index $r_h$')
    g.ax_joint.set_ylabel('Rank retweets $r_{rt}$')

    g.ax_marg_x.tick_params(axis='x', direction='out', which='major', size=0)
    g.ax_marg_y.tick_params(axis='y', direction='out', which='major', size=0)

    g.ax_marg_y.set_xlim((None, g.ax_marg_y.get_xlim()[1]*1.1))
    g.ax_marg_x.set_ylim((None, g.ax_marg_x.get_ylim()[1]*1.1))

    # log scale
    if do_log:
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')

    # diagonal line
    g.ax_joint.plot([0, num_ranks], [0, num_ranks], color='0.15', ls='dashed', lw=.5)

    # num ticks
    num_ticks = 5
    g.ax_joint.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    g.ax_joint.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

    save_fig(g, 'fig4a', 1, dpi=600, plot_formats=['png', 'pdf'])

def bootstrap_ci(df, cols, n_iterations=1000, sample_frac=.1, agg='mean', alpha=.95):
    statistics = []
    for _ in range(n_iterations):
        sample = df.sample(int(sample_frac*len(df)), replace=True)
        stats = sample[cols].agg(agg)
        statistics.append({c: stats[c] for c in cols})
    df_stats = pd.DataFrame(statistics)
    p = (1-alpha)/2
    upper = df_stats.quantile(1 - p)
    lower = df_stats.quantile(p)
    ci = {c: [lower[c], upper[c]] for c in cols}
    return ci

def compute_ci_new(df, n_iterations=10, alpha=.95):
    stats = pd.DataFrame()
    cols = ['h_index_rank', 'num_retweets_rank']
    for _ in range(n_iterations):
        sample = df.sample(4000, replace=True)
        sample['h_index_rank'] = sample.h_index.rank(method='first', ascending=False).astype(int)
        sample['num_retweets_rank'] = sample.num_retweets.rank(method='first', ascending=False).astype(int)
        _stats = sample.groupby('super_community')[cols].agg('mean')
        stats = pd.concat([stats, _stats.reset_index()])
    p = (1-alpha)/2
    ci = {}
    for sc, grp in stats.groupby('super_community'):
        upper = grp.quantile(1-p)
        lower = grp.quantile(p)
        ci[sc] = {c: [lower[c], upper[c]] for c in cols}
    return ci

@plot
def fig4b():
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {'International expert': palette[4], 'National expert': palette[1], 'Political': palette[2], 'Other': '.5'}
    df = read_data()
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=False, sharey=False)

    for _t, grp in df.groupby('super_community'):
        means = grp.mean()
        ci = bootstrap_ci(grp, ['h_index_rank', 'num_retweets_rank'])
        for k, v in ci.items():
            ci[k] = [[abs(means[k] - v[0])], [abs(means[k] - v[1])]]
        ax.errorbar(means['h_index_rank'], means['num_retweets_rank'], xerr=ci['h_index_rank'], yerr=ci['num_retweets_rank'], label=_t, c=color_dict[_t], ms=5, marker='.', capsize=.8, elinewidth=.5, markeredgewidth=.5)

    num_ranks = len(df)
    ax.set_xlim((num_ranks, 0))
    ax.set_ylim((num_ranks, 0))
    ax.grid(True)

    # num ticks
    num_ticks = 4
    ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

    # annotate
    len_arr = 500
    center = num_ranks/2
    arrow_props = dict(fc='.15', shrink=.1, width=.15, headwidth=2, headlength=2)
    ax.annotate('', xy=(center-len_arr, center+len_arr), xytext=(center, center), ha='left', va='top', arrowprops=arrow_props)
    ax.annotate('', xy=(center+len_arr, center-len_arr), xytext=(center, center), ha='left', va='top', arrowprops=arrow_props)
    ax.text(center+len_arr, center-len_arr, 'Over-ranked\nby retweets\n(high virality)', ha='right', va='bottom', fontsize=6)
    ax.text(center-len_arr, center+len_arr, 'Under-ranked\nby retweets\n(low virality)', ha='left', va='top', fontsize=6)

    # axis labels
    ax.set_xlabel(r'Avg. rank $h$-index $r_h$')
    ax.set_ylabel('Avg. rank retweets $r_{rt}$')

    sns.despine()
    ax.set_aspect('equal')

    # diagonal line
    ax.plot([0, num_ranks], [0, num_ranks], color='0.15', ls='dashed', lw=.5, zorder=0)

    save_fig(fig, 'fig4b', 1, dpi=600, plot_formats=['png', 'pdf'])


if __name__ == "__main__":
    # fig4a()
    fig4b()
