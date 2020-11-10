import pandas as pd
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from helpers.analysis_helpers import save_fig, plot
import scipy.stats
import matplotlib.transforms
import matplotlib.gridspec
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR = os.path.join('..', 'data', 'fig2')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')

nicer_category_names = {
        'art': 'Arts and Entertainment',
        'business': 'Business',
        'healthcare': 'Healthcare',
        'media': 'Media',
        'ngo': 'NGO',
        'other': 'Other',
        'political_supporter': 'Political Supporter',
        'politics': 'Politics & Government',
        'adult_content': 'Adult content',
        'public_services': 'Public Services',
        'religion': 'Religion',
        'science': 'Science',
        'sports': 'Sports'
        }

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


def get_category_data(without_other=True):
    f_path = os.path.join(DATA_DIR, 'user_category_data_plot.parquet')
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f'File {f_path} could not be found. Check code in get_categories.py to recompute.')
    df = pd.read_parquet(f_path)
    df.index.name = 'categories'
    df.index = df.reset_index().categories.apply(lambda s: nicer_category_names[s])
    if without_other:
        df = df.drop(index='Other')
    return df

def get_geo_entropy():
    f_path = os.path.join(DATA_DIR, 'community_geo_data.parquet')
    if not os.path.isfile(f_path):
        raise FileNotFoundError(f'File {f_path} could not be found. Check code in get_categories.py to recompute.')
    df = pd.read_parquet(f_path)
    df = pd.DataFrame(scipy.stats.entropy(df, axis=1), index=df.index)
    df = df.rename(columns={0: 'Internationality'})
    return df

@plot
def main():
    df = get_category_data(without_other=False)
    df_entropy = get_geo_entropy()

    df = df.T
    df = df.divide(df.sum(axis=1), axis=0)
    df.columns = df.columns.tolist()
    df['Internationality'] = df_entropy['Internationality']
    df = df[[c for c in df.columns if c != 'Other']]
    max_val = df.subtract(df.mean(axis=0), axis=1).divide(df.std(axis=0), axis=1).max().max()

    usecols = ['Political Supporter', 'Public Services', 'Politics & Government',  'Media', 'Healthcare', 'Science', 'Internationality']
    df = df[usecols]

    width = 2.2
    height = 3.7
    # cmap = None
    cmap = 'RdBu_r'
    # cmap = 'Blues'
    # cmap = 'Reds'

    palette = [c['color'] for c in matplotlib.rcParams['axes.prop_cycle']]
    colors = {'International expert': palette[4], 'National expert': palette[1], 'Political': palette[2], 'Other': '.5'}
    df.index.name = 'communities'
    super_communities = df.reset_index().communities.apply(lambda s: community_to_types[s]).copy()
    super_community_colors = super_communities.apply(lambda s: colors[s])

    g = sns.clustermap(df, dendrogram_ratio=.3, colors_ratio=.04, cmap=cmap, center=0, lw=0.4, col_cluster=False, z_score=1,
            figsize=(width, height), cbar_pos=(1.1, .33, .02, .2), row_colors=super_community_colors)

    fig = plt.gcf()
    fig.delaxes(g.ax_col_dendrogram)

    g.ax_row_colors.get_xaxis().set_visible(False)

    # colorbar formatting
    g.ax_cbar.set_ylabel('Z-score', rotation=0, ha='left', va='center')
    g.ax_cbar.locator_params(nbins=5)
    g.ax_cbar.tick_params(length=2, axis='both', which='major', direction='out')

    # create type legend
    type_legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    labels = list(colors.keys())
    labels[labels.index('National expert')] = 'National elite'
    legend_type = g.ax_heatmap.legend(labels=labels, loc='center left', bbox_to_anchor=(1.3, .85),
            handles=type_legend_patches, frameon=False, title='Super-community', handlelength=.6, handleheight=1)

    # move axis labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=75, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.ax_row_colors.set_xticklabels(g.ax_row_colors.get_xticklabels(), rotation=75, ha='right')
    offset = matplotlib.transforms.ScaledTranslation(.05, 0, fig.dpi_scale_trans)
    for ax in [g.ax_heatmap, g.ax_row_colors]:
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

    # other cosmetics
    g.ax_heatmap.tick_params(axis='both', which='both', bottom=False, right=False)
    g.ax_row_colors.tick_params(axis='both', which='both', bottom=False, right=False)
    g.ax_heatmap.set_ylabel('Community')

    save_fig(fig, 'fig2', 1, plot_formats=['png', 'pdf'], dpi=600)

if __name__ == "__main__":
    main()
