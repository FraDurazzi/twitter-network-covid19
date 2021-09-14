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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

DATA_DIR = os.path.join('..', 'data', 'fig4')

community_to_types = {
        'A': 'Other',
        'B': 'International sci-health',
        'C': 'Political',
        'D': 'National elite',
        'E': 'Other',
        'F': 'Political',
        'G': 'International sci-health',
        'H': 'Political',
        'I': 'National elite',
        'J': 'National elite',
        'K': 'Political',
        'L': 'Political',
        'M': 'National elite',
        'N': 'Other',
        'O': 'Other'}

def get_interaction_data():
    # interaction data
    f_path = os.path.join(DATA_DIR, 'retweet_retweeted_community_data_agg.parquet')
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f'File {f_path} could not be found. Check code in network_vs_time.py to recompute.')
    df = pd.read_parquet(f_path)
    df['out_type'] = df.community_out.apply(lambda s: community_to_types[s])
    df['in_type'] = df.community_in.apply(lambda s: community_to_types[s])
    df_counts = []
    for date, grp in df.groupby('date'):
        total = grp.counts.sum()
        by_type = {}
        # total received tweets by type
        for comm_type, grp_comm in grp.groupby('in_type'):
            by_type[f'received_cluster_links_{comm_type}'] = grp_comm.counts.sum()
        # total sent tweets by type
        for comm_type, grp_comm in grp.groupby('out_type'):
            by_type[f'sent_cluster_links_{comm_type}'] = grp_comm.counts.sum()
        # fraction of intra-tweets received by type
        for comm_type, grp_intra in grp[grp['in_type'] == grp['out_type']].groupby('in_type'):
            by_type[f'intra_cluster_links_{comm_type}'] = grp_intra.counts.sum() / total
        # fraction of inter-tweets received by type
        for comm_type, grp_inter in grp[grp['in_type'] != grp['out_type']].groupby('in_type'):
            by_type[f'inter_cluster_links_{comm_type}'] = grp_inter.counts.sum() / total
        df_counts.append({'date': date, 'total': total, **by_type})
    df_counts = pd.DataFrame(df_counts)
    df_counts = df_counts.set_index('date')
    return df_counts

def get_community_size_data():
    # community size data
    f_path = os.path.join(DATA_DIR, 'retweet_retweeted_community_data_size.parquet')
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f'File {f_path} could not be found. Check code in network_vs_time.py to recompute.')
    df = pd.read_parquet(f_path)
    df = df.reset_index().melt(id_vars=['date'], var_name='community')
    df['type'] = df.community.apply(lambda s: community_to_types[s])
    df = df.groupby(['date', 'type']).sum().reset_index().set_index('date').pivot(columns='type', values='value')
    return df

def label_subplots(axes, upper_case=True, offset_points=(-20, 0), start_ord=67):
    for ax, lab in zip(axes, ['{}'.format(chr(j)) for j in range(start_ord, start_ord + len(axes))]):
        ax.annotate(lab, (0, 1), xytext=offset_points, xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', weight='semibold')

@plot
def main():
    # read data
    df_int = get_interaction_data()
    df_size = get_community_size_data()
    df_size = df_size[df_int.index.min():df_int.index.max()]
    df_size['Total'] = df_size.sum(axis=1)
    dfs = {}
    for _type in ['inter_cluster', 'intra_cluster', 'received']:
        df = df_int[df_int.columns[df_int.columns.str.startswith(_type)]].copy()
        col_rename = {}
        for col in df.columns:
            col_rename[col] = col.split('_')[-1]
        df = df.rename(columns=col_rename)
        df['Total'] = df.sum(axis=1)
        dfs[_type] = df

    col_order = ['Total', 'International sci-health', 'National elite', 'Political', 'Other']
    palette = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]
    # orange, blue,
    color_dict = {'International sci-health': palette[4], 'National elite': palette[1], 'Political': palette[2], 'Other': '.5', 'Total': '0.15'}
    ls_dict = {'International sci-health': '-', 'National elite': '-', 'Political': '-', 'Other': '-', 'Total': 'dotted'}
    m_dict = {'International sci-health': '.', 'National elite': '.', 'Political': '.', 'Other': '.', 'Total': None}
    z_order = {'International sci-health': 4, 'National elite': 3, 'Political': 2, 'Other': 1,  'Total': 0}
    colors = [color_dict[c] for c in col_order]
    col_order_rev = col_order[::-1]
    colors_rev = [color_dict[c] for c in col_order_rev]

    # plot
    fig, all_axes = plt.subplots(2, 2, figsize=(4.5, 3.5), sharex=False, sharey=False)

    # specs
    lw = 1
    ec = 'white'
    ms = 5

    # panels A
    ax = all_axes[0][0]
    for col in df[col_order]:
        ax.plot(df_size.index.values, df_size[col].values, label=col, color=color_dict[col], marker=m_dict[col], ms=ms, lw=lw, ls=ls_dict[col], zorder=z_order[col])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel('$N$')
    ax.set_title('Number of users', fontsize=7)
    ax.set_ylim((0, 0.6*10**7))
    ax.grid()

    # panels B
    ax = all_axes[0][1]
    df = dfs['received']
    df /= df_size # normalize by size of type
    for col in df[col_order]:
        ax.plot(df.index.values, df[col].values, label=col, color=color_dict[col], marker=m_dict[col], ms=ms, lw=lw, ls=ls_dict[col], zorder=z_order[col])
    ax.set_ylabel('$A_u$')
    ax.set_title('Avg. attention per user', fontsize=7)
    ax.set_ylim((0, 8))
    ax.grid()

    # panels C
    df = dfs['inter_cluster']
    ax = all_axes[1][0]
    for col in df[col_order]:
        ax.plot(df.index.values, df[col], label=col, color=color_dict[col], marker=m_dict[col], ms=ms, lw=lw, ls=ls_dict[col], zorder=z_order[col])

    ax.set_title('External attention component', fontsize=7)
    ax.set_ylabel(r'$a^{ext}$')
    ax.set_xlabel('')
    ax.grid()
    ax.set_ylim((0, .35))

    # panels D
    df = dfs['intra_cluster']
    ax = all_axes[1][1]
    for col in df[col_order]:
        ax.plot(df.index.values, df[col], label=col, color=color_dict[col], marker=m_dict[col], ms=ms, lw=lw, ls=ls_dict[col], zorder=z_order[col])

    ax.set_title('Internal attention component', fontsize=7)
    ax.set_ylabel(r'$a^{int}$')
    ax.set_xlabel('')
    ax.grid()
    ax.set_ylim((0, .95))

    # vertical lines
    for i, time_pos in enumerate([2, 9]):
        for ax_row in all_axes:
            for ax in ax_row:
                ax.axvline(df.index[time_pos], ls='dashed', lw=.5, c='.15', zorder=0)
                ax.annotate(chr(97 + i), (df.index[time_pos], ax.get_ylim()[1]*.95), xytext=(2, 0), textcoords='offset points', ha='left', va='center')

    # tick style
    for ax_row in all_axes:
        for ax in ax_row:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.tick_params(axis='x', direction='out', which='minor', size=2)
            ax.tick_params(axis='x', direction='out', which='major', size=2)
            ax.tick_params(axis='y', which='major', direction='out', size=2)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(hspace=.5)
    save_fig(plt.gcf(), 'fig4', 1, dpi=600, plot_formats=['png', 'pdf'])

if __name__ == "__main__":
    main()
