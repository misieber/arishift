import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as pp
import matplotlib.patches as patches

import locale
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

month_abbrs = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
month_abbrs_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
len_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monthcats = pd.Categorical(month_abbrs, categories=month_abbrs, ordered=True)
months = dict([mz for mz in zip(month_abbrs, len_months)])

causes = {'Insgesamt': ['Todesursachen insgesamt', False],
          'A00-B99': ['Parasitär', False],          
          'E00-E90': ['Stoffwechsel', False],
          'E10-E14': ['Diabetes', False],
          'F00-F99': ['Psyche und Verhalten', False],
          'G00-G99': ['Nervensystem', False],
          'I00-I99': ['Cardiovascular diseases', True], 
          'C00-D48': ['Cancer', False],
          'C15-C26': ['Digestive cancers', False],
          'C30-C39': ['Respiratory cancers', False],
          'C43-C44': ['Skin cancer', False],
          'C50': ['Breast cancer', False],
          'C81-C96': ['Lymphatic and hema', False],
          'J00-J99': ['Respiratory', False],
          'J20-J22': ['Respiratory, not influenza', False],
          'U071': ['Covid', False],
          'J09-J18': ['seasonal influenza', False],
          'K00-K93': ['Verdauungssystem', False],
          'N00-N99': ['Urogenitalsystem', False],
          'R00-R99': ['Abnorm, nicht klassifiziert', False],
          'X60-X84': ['Suicide', False],
          'V01-V99': ['Transport', False],
          'X85-Y09': ['Angriff', False],
          'W00-W19': ['Sturz', False],
          'R95': ['Kindstot', False],
          'P00-P96': ['Perinatal', False],
          'Y10-Y34': ['Unknown', False],
          'F10-F19': ['Alcohol and psychrotopic substances', False],
          'K70-K77': ['Liver', False],
          'V01-Y': ['test', False],
          'RESPSUM': ['Respiratory diseases', True]
          }

df_raw_causes = pd.read_csv('data/Destatis_800472_monatliche_Todesursachenstatistik_2009-2022.csv', sep=',')
df_raw_causes = df_raw_causes[[True if m in months else False for m in df_raw_causes['month']]]

df_raw_causes['month'] = pd.Categorical(df_raw_causes['month'], categories=months.keys(), ordered=True)
df_raw_causes['value'] = pd.to_numeric(df_raw_causes['value'], errors='coerce').fillna(0)
df_raw_causes['value'] = [row['value']/months[row['month']] for rowi, row in df_raw_causes.iterrows()]

df_raw_causes['datetime'] = ['01-' + r['month'] + '-' + str(r['year']) for i, r in df_raw_causes.iterrows()]
df_raw_causes['datetime'] = pd.to_datetime(df_raw_causes['datetime'])

df_grp = df_raw_causes[(df_raw_causes['icd10_code'] == 'U071') | (df_raw_causes['icd10_code'] == 'U072') | (df_raw_causes['icd10_code'] == 'J00-J99')].groupby(['year', 'month', 'datetime'], as_index=False, observed=True)['value'].sum()
df_grp['icd10_code'] = ['RESPSUM']*df_grp.shape[0]
df_grp['icd10_desc'] = ['Summe U00-U49 und J00-J99']*df_grp.shape[0]

df_raw_causes = df_raw_causes.sort_values(['year', 'month'])
df_raw_causes = pd.concat([df_raw_causes, df_grp]).reset_index(drop=True)

df_raw_causes['season'] = [r['year']-1 if r['month'] < monthcats[[6]] else r['year'] for i, r in df_raw_causes.iterrows()]

seasons = df_raw_causes['season'].unique()
seasons = seasons[seasons > 2012]

fig, axs = pp.subplots(1, 2, figsize=(13, 5))

j=0
for cause in causes:
    if causes[cause][1]:

        df_plot = df_raw_causes[(df_raw_causes['icd10_code'] == cause) & (df_raw_causes['season'] >= seasons.min())].reset_index(drop=True)
        
        df_plot_pivot = df_plot.pivot(index='season', columns='month', values='value')
        df_plot_pivot = df_plot_pivot[list(df_plot_pivot.columns[6:]) + list(df_plot_pivot.columns[:6])]

        sns.heatmap(df_plot_pivot, ax=axs[j], cmap=sns.cubehelix_palette(start=1, rot=0, dark=0.5, light=1, gamma=3, as_cmap=True), square=True, cbar=True, cbar_kws={'label':'incedence', 'location': 'right', 'shrink': 0.5, 'format': '%.0f'})

        for i, row in enumerate(df_plot_pivot.iterrows()):
            maxmonth = row[1].argmax()
            maxmarker = patches.Rectangle((maxmonth+0.01, i+0.01), 0.99, 0.99, rotation_point='center', linewidth=0, edgecolor='black', facecolor='none', hatch='//', alpha=0.99)
            axs[j].add_patch(maxmarker)
        
        axs[j].set_xlabel('')
        axs[j].set_xticks(np.arange(len(df_plot_pivot.columns))+0.5, month_abbrs_EN[6:] + month_abbrs_EN[:6], fontsize=12)
        axs[j].tick_params(axis='x', length=5, color='white')

        seasonticks = [str(season) + '/' + str(season+1)[-2:] for season in seasons]        
        axs[j].set_ylabel('')
        axs[j].set_yticks(np.arange(len(df_plot_pivot.index)) + 0.6, seasonticks, fontsize=12)
        axs[j].tick_params(axis='y', length=5, color='white')
        
        axs[j].set_title(causes[cause][0], pad=20, fontsize=16)
        
        j+=1
  
pp.tight_layout()
pp.savefig('figures/figure_4.tiff', bbox_inches='tight', dpi=600)

from PIL import Image
im = Image.open('figures/figure_4.tiff')
im.save('figures/figure_4.tiff', compression='tiff_lzw')