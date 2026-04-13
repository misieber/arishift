import pandas as pd
import numpy as np
from matplotlib import pyplot as pp
import calendar

plt_data_sources = ['sri', 'ari', 'sari', 'mort']
plt_age_group = 'all'

first_season = '2011/12'
last_season = '2024/25'
incidence_level = 100000
season_start_week = 26
season_col_name = 'season'
year_col_name = 'Jahr'
time_col_name = 'Kalenderwoche'
week_col_name = 'week'
incidence_col_name = 'Inzidenz'
covid_incidence_col_name = 'Covid_Inzidenz'
output_dir = 'figures/'


plt_txts = {'sri': {'plt_title': 'self-reported respiratory infections', 'main_label': 'symptomatic respiratory infections', 'covid_label': ''},
              'ari': {'plt_title': 'acute respiratory infections', 'main_label': 'main diagnosis ARI (ICD-10 codes J00-J22)', 'covid_label': r'$\bf{and}$' + ' diagnosis COVID-19 (ICD-10 code U07.1)'},
              'sari': {'plt_title': 'severe acute respiratory infections', 'main_label': 'main diagnosis SARI (ICD-10 codes J09-J22)', 'covid_label': r'$\bf{and}$' + ' diagnosis COVID-19 (ICD-10 code U07.1)'},
              'mort': {'plt_title': 'all-cause mortality', 'main_label': 'reported deaths', 'covid_label': ''}}


raw_age_grps = {
                'all' : {'ari': '00+', 'sari': '00+', 'mort': 'Insgesamt', 'covid_ari': '00+', 'covid_sari': '00+', 'sri': '00+'},
                'low': {'ari': '0-4', 'sari': '0-4', 'mort': '0-29', 'covid_ari': '00-04', 'covid_sari': '', 'sri': '0-4'},
                'med': {'ari': '35-59', 'sari': '35-59', 'mort': '45-49', 'covid_ari': '35-59', 'covid_sari': '', 'sri': '35-59'},
                'high': {'ari': '60+', 'sari': '80+', 'mort': '80-84', 'covid_ari': '60+', 'covid_sari': '80+', 'sri': '60+'},
                }


#### SARI #####
df_raw_sari = pd.read_csv('data/SARI-Hospitalisierungsinzidenz.tsv', sep='\t')
df_raw_sari = df_raw_sari.rename(columns={'Hospitalisierungsinzidenz': incidence_col_name})
df_raw_sari = df_raw_sari[df_raw_sari['SARI'] == 'Gesamt']
df_raw_sari = df_raw_sari[df_raw_sari['Altersgruppe'] == raw_age_grps[plt_age_group]['sari']]
df_raw_sari[week_col_name] = df_raw_sari[time_col_name]

#### ARI #####
df_raw_ari = pd.read_csv('data/ARE-Konsultationsinzidenz.tsv', sep='\t')
df_raw_ari = df_raw_ari.rename(columns={'ARE_Konsultationsinzidenz': incidence_col_name})
df_raw_ari = df_raw_ari[df_raw_ari['Bundesland'] == 'Bundesweit']
df_raw_ari = df_raw_ari[df_raw_ari['Altersgruppe'] == raw_age_grps[plt_age_group]['ari']]
df_raw_ari[week_col_name] = df_raw_ari[time_col_name]

#### sri #####
df_raw_flu = pd.read_csv('data/GrippeWeb_Daten_des_Wochenberichts.tsv', sep='\t')
df_raw_flu = df_raw_flu.rename(columns={'Inzidenz': incidence_col_name})
df_raw_flu = df_raw_flu[df_raw_flu['Region'] == 'Bundesweit']
df_raw_flu = df_raw_flu[df_raw_flu['Erkrankung'] == 'ARE']
df_raw_flu = df_raw_flu[df_raw_flu['Altersgruppe'] == raw_age_grps[plt_age_group]['sri']]
df_raw_flu[week_col_name] = df_raw_flu[time_col_name]

#### MORT #####
df_raw_mort = pd.read_csv('data/csv-12613-02.csv', sep=',')
df_raw_mort_current = pd.read_csv('data/csv-12613-01.csv', sep=',')

df_raw_mort = pd.concat([df_raw_mort, df_raw_mort_current])
df_raw_mort = df_raw_mort[df_raw_mort['Alter'] == raw_age_grps[plt_age_group]['mort']]

df_raw_mort = df_raw_mort.rename(columns={'Sterbefaelle': incidence_col_name})

df_raw_mort[incidence_col_name] = df_raw_mort[incidence_col_name].replace(['X', 'X ', '...'], 0)
df_raw_mort[incidence_col_name] = df_raw_mort[incidence_col_name].astype(int)
df_raw_mort[incidence_col_name] = df_raw_mort[incidence_col_name]
df_raw_mort[year_col_name] = df_raw_mort[year_col_name].astype(int)
df_raw_mort[time_col_name] = df_raw_mort[time_col_name].astype(int)
df_raw_mort[week_col_name] = [str(row[year_col_name]) + '-W{:02d}'.format(row[time_col_name]) for idx, row in df_raw_mort.iterrows()]
df_raw_mort = df_raw_mort[df_raw_mort[incidence_col_name] > 0]

df_demo = pd.read_csv('data/12411-0005.csv', sep=',')
df_demo.set_index(['age'], inplace=True)

agelims = raw_age_grps[plt_age_group]['mort'].split('-')
if len(agelims) == 2:
    df_demo = df_demo.loc[agelims[0]:agelims[1]]

data_start_year = df_raw_mort[year_col_name].unique()[0]

df_demo.columns = df_demo.columns.map(int)
pop = df_demo.sum()/incidence_level
pop = pop[pop.index >= data_start_year]

yweeks = len(df_raw_mort[year_col_name].unique())*[0]
for i, (yi, y) in enumerate(df_raw_mort.groupby(year_col_name)):
    yweeks[i] = len(y)

df_raw_mort[incidence_col_name] = (df_raw_mort[incidence_col_name].values/np.repeat(pop, yweeks)).values


# merge datasets for plotting
df_plot = df_raw_mort.merge(df_raw_ari, how='left', on='week', suffixes=['_mort', '_ari'])
df_plot = df_plot.merge(df_raw_sari, how='left', on='week', suffixes=['_ari', '_sari'])
df_plot = df_plot.rename(columns={incidence_col_name: incidence_col_name + '_sari'})
df_plot = df_plot.merge(df_raw_flu, how='left', on='week', suffixes=['', '_sri'])
df_plot = df_plot.rename(columns={incidence_col_name: incidence_col_name + '_sri'})

df_plot = df_plot.filter(['week', 'Inzidenz_sri', 'Inzidenz_ari', 'Inzidenz_sari', 'Inzidenz_mort'])

df_plot['season'] = ''

for i, w in enumerate(df_plot[week_col_name]):
    year = w[:4]
    week = w[6:]
    if int(week) <= season_start_week:
        df_plot.loc[i, 'season'] = str(int(year)-1) + '/' + year[2:]
    else:
        df_plot.loc[i, 'season'] = year + '/' + str(int(year[2:])+1)

df_plot = df_plot[(df_plot['season'] >= first_season) & (df_plot['season'] <= last_season)]
df_plot = df_plot.reset_index()

### timeseries plot ###
seasons = np.array(df_plot[season_col_name].unique())

nseasons_pre = sum(seasons < '2020/21')
nseasons_pan = sum(seasons >= '2020/21')

colors_pre = pp.cm.coolwarm(np.linspace(1/3, 0, nseasons_pre))
colors_pan = pp.cm.coolwarm(np.linspace(1, 2/3, nseasons_pan))
colors = np.concatenate([colors_pre,colors_pan])

xrange = df_plot[week_col_name].values
w01_pos = [(i, d[:4]) for i, d in enumerate(df_plot[week_col_name]) if 'W01' in d]

fig, axs = pp.subplots(2,2, figsize=(12, 6), sharex='all')
for axi, data_src in enumerate(plt_data_sources):

    ax = axs[int(axi/2), axi%2]
    
    ax.set_title('(' + chr(axi+97) + ') ' + plt_txts[data_src]['plt_title'], loc='left')

    for w01x in w01_pos:
        ax.axvline(w01x[0], color='lightgrey')

    for i, season in enumerate(seasons):
        season_data = df_plot[df_plot[season_col_name] == season]
        
        if i < len(seasons)-1:
            season_data = pd.concat([season_data, df_plot[df_plot[season_col_name] == seasons[i+1]].iloc[0:1]])
        
        ax.plot(season_data.index.values, season_data[incidence_col_name + '_' + data_src].values, ls='-', lw=3, color=colors[i], label=plt_txts[data_src]['main_label'])

    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    ax.set_xticks([w[0] for w in w01_pos], [w[1] for w in w01_pos], fontsize=8)

    ax.set_ylabel(r'incidence (cases/' + str(incidence_level) + ')' + '\n', fontsize='large')

pp.tight_layout()
pp.subplots_adjust(wspace=0.2, hspace=0.5)
pp.savefig(output_dir + 'figure_1.pdf')


###### polar plot ######
fig, axs = pp.subplots(2,2, figsize=(8, 7.5), subplot_kw=dict(projection='polar'))

for axi, data_src in enumerate(plt_data_sources):
    
    ax = axs[int(axi/2), axi%2]
    
    ax.set_title('(' + chr(axi+97) + ') ' + plt_txts[data_src]['plt_title'] + '\n', loc='left')

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')

    ym = df_plot[incidence_col_name + '_' + data_src].fillna(0).values.max()
    yl = 1.9*ym

    min_seasonlng = 52
    for i, season in enumerate(seasons):
        season_data = df_plot[df_plot[season_col_name] == season]
        season_data.reset_index(drop=True, inplace=True)
        
        if season_data[incidence_col_name + '_' + data_src].isnull().all():
            continue
        
        weeks_in_season = season_data.shape[0]
        theta = np.linspace(season_start_week/weeks_in_season*2*np.pi, season_start_week/weeks_in_season*2*np.pi + 2*np.pi, weeks_in_season)
        if weeks_in_season < min_seasonlng:
            theta = np.linspace(season_start_week/min_seasonlng*2*np.pi, season_start_week/min_seasonlng*2*np.pi + weeks_in_season/min_seasonlng*2*np.pi, weeks_in_season)
        
        ax.plot(theta, season_data[incidence_col_name + '_' + data_src].values, ls='-', lw=1.5, color=colors[i], alpha=0.8, label=season)
        
        max_n = 4
        if season_data[incidence_col_name + '_' + data_src].count() >= max_n:
            top_i = np.argpartition(season_data[incidence_col_name + '_' + data_src].fillna(-1), -max_n)[-max_n:]
            peak_weeks = season_data.iloc[top_i]
            max_incidence = peak_weeks[incidence_col_name + '_' + data_src].values.max()
            min_incidence = peak_weeks[incidence_col_name + '_' + data_src].values.min()
            mean_incidence = np.mean(peak_weeks[incidence_col_name + '_' + data_src].values)
            
            for j, peak_week in peak_weeks.iterrows():
                wr = (1.07 + 0.06*i)*ym
                wll = 0.51
                aa = np.linspace(theta[j]-wll/min_seasonlng*2*np.pi, theta[j]+wll/min_seasonlng*2*np.pi, 5)        
                
                peak_thresh = min_incidence/mean_incidence
                week_incidence = season_data[incidence_col_name + '_' + data_src].values[j]

                ax.plot(aa, [wr for _ in aa], linewidth=3, color=colors[i], alpha=0.8, solid_capstyle = 'butt')
        

    ax.set_xticks(np.linspace(1/2*1/12*2*np.pi, 1/2*1/12*2*np.pi + 11/12*2*np.pi, 12), calendar.month_abbr[1:])
    ax.xaxis.set_tick_params(labelsize='small', grid_alpha=0)

    theta = np.linspace(0, 2*np.pi, 13)
    ax.vlines(theta[:-1], 0, yl, linewidth=1, color='grey', alpha=0.3)

    ax.set_rlabel_position(180)

    ax.set_ylim(0, yl)

    yti = ax.get_yticks()
    ax.yaxis.set_ticks([_ for _ in yti if _ < ym])

    ax.yaxis.set_tick_params(labelsize='x-small', grid_alpha=0.5)

    ax.spines['polar'].set_visible(False)
    
h, l = axs.flat[np.argmax([len(ax.get_legend_handles_labels()[1]) for ax in axs.flat])].get_legend_handles_labels()

fig.legend(handles=h, labels=l, loc='center left', bbox_to_anchor=(0.95, 0.5), frameon=False, markerscale=20, fontsize='small', title='season')

pp.tight_layout()
pp.subplots_adjust(wspace=0, hspace=0.5)
pp.savefig(output_dir + 'figure_2.pdf', bbox_inches='tight')
