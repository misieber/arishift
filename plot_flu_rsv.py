import pandas as pd
import numpy as np
from matplotlib import pyplot as pp
import seaborn as sns

df_raw_flu = pd.read_csv('data/survstat_flu_y27.csv', sep=',')
df_raw_flu = df_raw_flu.rename(columns={'Unnamed: 0': 'season'})
df_raw_flu = df_raw_flu[df_raw_flu['season'] < '2025/26']
df_raw_flu = df_raw_flu.iloc[:,:52].fillna(0).join(df_raw_flu.iloc[:,52:])
df_raw_flu = pd.melt(df_raw_flu, id_vars='season', var_name='week', value_name='flu')
df_raw_flu = df_raw_flu.set_index(['season', 'week'])

df_raw_rsv = pd.read_csv('data/survstat_rsv_y27.csv', sep=',')
df_raw_rsv = df_raw_rsv.rename(columns={'Unnamed: 0': 'season'})
df_raw_rsv = df_raw_rsv[df_raw_rsv['season'] < '2025/26']
df_raw_rsv = pd.melt(df_raw_rsv, id_vars='season', var_name='week', value_name='rsv')
df_raw_rsv = df_raw_rsv.set_index(['season', 'week'])

df_raw_covid = pd.read_csv('data/survstat_covid_y27.csv', sep=',')
df_raw_covid = df_raw_covid.rename(columns={'Unnamed: 0': 'season'})
df_raw_covid = df_raw_covid[df_raw_covid['season'] < '2025/26']
df_raw_covid = pd.melt(df_raw_covid, id_vars='season', var_name='week', value_name='covid')
df_raw_covid = df_raw_covid.set_index(['season', 'week'])

df_plot = df_raw_flu.join(df_raw_rsv)
df_plot = df_plot.join(df_raw_covid)
df_plot = df_plot.reset_index()
df_plot = df_plot[df_plot['season'] > '2003/04']

pan_start = '2020/21'

cm = ['#ff7f0e', '#2ca02c', 'lightblue']

seasons = df_plot['season'].unique()

month_abbrs_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

df_plot_flu = df_plot.pivot(index='season', columns='week', values='flu')
df_plot_rsv = df_plot.pivot(index='season', columns='week', values='rsv')
df_plot_covid = df_plot.pivot(index='season', columns='week', values='covid')

df_plot_flu = df_plot_flu.fillna(0)
df_plot_flu = df_plot_flu.reset_index(drop=False)
df_plot_flu.index = range(0, 3*len(df_plot_flu), 3)

df_plot_rsv = df_plot_rsv.fillna(0)
df_plot_rsv = df_plot_rsv.reset_index(drop=False)
df_plot_rsv.index = range(1, 3*len(df_plot_rsv), 3)

df_plot_covid = df_plot_covid.fillna(0)
df_plot_covid = df_plot_covid.reset_index(drop=False)
df_plot_covid.index = range(2, 3*len(df_plot_covid), 3)

df_c = pd.concat([df_plot_flu, df_plot_rsv, df_plot_covid])
df_c = df_c.sort_index()

df_c = df_c.set_index('season')

cmaps = [0, 1, 2]
titles = ['Influenza', 'RSV', 'COVID']

h_ratios = [21, 13, 7]

fig, axs = pp.subplots(nrows=3, ncols=1, height_ratios=h_ratios, figsize=(8,8))
sns.set(font_scale=0.8)

for i, cause in enumerate(['flu', 'rsv', 'covid']):

    df_plot_pivot = df_plot.pivot(index='season', columns='week', values=cause)
    df_plot_pivot = df_plot_pivot.fillna(0)
    df_plot_pivot = df_plot_pivot.iloc[(df_plot_pivot.sum(axis=1) > 0).argmax():,:]
    
    shrink = 0.75*min(h_ratios)/h_ratios[i]
    sns.heatmap(df_plot_pivot, ax=axs[i], cmap=sns.cubehelix_palette(start=cmaps[i], rot=0, dark=0.5, light=1, gamma=3, as_cmap=True), square=True, cbar=True, cbar_kws={'label':'incedence', 'location': 'right', 'ticks': [0, df_plot_pivot.max().max()], 'shrink': shrink, 'format': '%.0f'})
    
    axs[i].set_xlabel('')
    
    xticklocs = np.linspace(0, 52, num=13)
    axs[i].set_xticks(xticklocs, [])
    axs[i].set_xticks(np.linspace(2.2, 50.1, num=12), minor=True)
    axs[i].set_xticklabels(month_abbrs_EN[6:] + month_abbrs_EN[:6], minor=True, fontsize=12)
    axs[i].tick_params(axis='x', which='minor', color='white')  
    axs[i].set_ylabel('')
    axs[i].set_title(titles[i], pad=10, fontsize=12)
    
pp.tight_layout()
pp.savefig('figures/figure_3.tiff', bbox_inches='tight', dpi=600)

from PIL import Image
im = Image.open('figures/figure_3.tiff')
im.save('figures/figure_3.tiff', compression='tiff_lzw')