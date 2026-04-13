import numpy as np
import matplotlib.pyplot as pp
import calendar
import matplotlib.colors as mcol
import  multiprocessing as mp
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib import text as mtext
import seaborn as sns
from matplotlib.patches import ArrowStyle
import math

from sir_model import solver

rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'STIXGeneral:italic'
rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used



ensemble_size = 1
p_range = np.linspace(0.3, 1, ensemble_size)
pool = mp.Pool(mp.cpu_count())
psols = pool.imap(solver, p_range, mp.cpu_count())
sols = [s for s in psols if s['success']]

example_p = 0.3
i_example = np.abs(p_range - example_p).argmin()
sol = sols[i_example]

sb_red = sns.color_palette("colorblind", as_cmap=True)[3]
sb_blue = sns.color_palette("colorblind", as_cmap=True)[0]
sb_green = sns.color_palette("colorblind", as_cmap=True)[2]
sb_grey = sns.color_palette("colorblind", as_cmap=True)[7]

fig = pp.figure(figsize=(10, 9))
# fig = pp.figure(figsize=(7.5, 7.5))

gs = gridspec.GridSpec(3, 2, height_ratios=[0.275, 0.275, 0.45])

# ax1 = fig.add_subplot(gs[0, 0])
# ax1.set_title('(a) transmission rate and recurring epidemics', loc='left')

# ax2 = fig.add_subplot(gs[0:2, 1], projection='polar')
# ax2.set_title(r'(b) prevalence, $I$', loc='left')

ax3 = fig.add_subplot(gs[1, 0])
# ax3.set_title('(c) effective reproduction number', loc='left')

# ax4 = fig.add_subplot(gs[2, :])
# ax4.set_title('(d) shift of epidemic season after NPIs', loc='left')

########## timeseries ########
# ax = ax1

# season_start = int(np.argmax(sol.y[1] > 2*I0) * t_step / 52)

# b = np.array([beta(t, p_range[i_example]) for t in sol.t])
# bmax = max(b)

# ax.plot(sol.t, sol.y[0], '-', lw=2, color=sb_green, label='S')
# ax.plot(sol.t, sol.y[1], lw=2, color=sb_red, label='I')

# ax.text(120, 0.12, r'$\mathbf{I}$', color=sb_red, fontweight='bold')
# ax.text(180, 0.83, r'$\mathbf{S}$', color=sb_green, fontweight='bold')
# ax.text(105, 0.9, r'$\mathbf{\beta}$', color=sb_grey, fontweight='bold')

# ax.set_ylabel('fraction of population')
# ax.set_ylim(0,1.05)

# # ax.set_xlabel('\nseason')
# ax.set_xticks(np.linspace(0, sol.t[-1], n_seasons + 1), [])
# ax.set_xticks(np.linspace(26, sol.t[-1]-26, n_seasons), minor=True)
# ax.set_xticklabels([str(i+1) for i in range(0,n_seasons)], minor=True)
# ax.tick_params(axis='x', which='minor', color='white')

# # ax.text((npi_season + 0.24)*season_length, 0.9, 'NPI')
# # ax.fill_betweenx([-1, 2], npi_season*season_length, (npi_season+1)*season_length + 0.5, lw=0, color='crimson', alpha=0.2)

# ax.text(0.15, -0.22, 'season', transform = ax.transAxes)

# ax.text(0.415, -0.22, 'NPI', transform = ax.transAxes)
# rect = patches.Rectangle((0.39, -0.005), 0.11, -0.25, linewidth=0, edgecolor='crimson', facecolor='crimson', alpha=0.2, transform = ax.transAxes, clip_on=False)
# ax.add_patch(rect)

# ax12 = ax1.twinx()
# ax12.plot(sol.t, b, lw=2, color=sb_grey, label=r'transmission rate $\beta$', zorder=1)
# ax12.set_ylim(0,1.1*bmax)
# ax12.set_ylabel(r'transmission rate $\beta$')

# ax.set_zorder(ax12.get_zorder()+1)
# ax.set_frame_on(False)
    
# ######### polar ########
# ax = ax2

# ax.set_theta_direction(-1)
# ax.set_theta_zero_location('S')

# t_plot = np.linspace(0, season_length, steps_per_season)
# theta = np.linspace(0, 2*np.pi, len(t_plot))

# # colors = list([pp.cm.Reds(0.9)]) + [pp.cm.Reds(0.5) if i == npi_season else pp.cm.Blues(c) for i,c in enumerate(np.linspace(0.5, 1, n_seasons-1))]
# colors = [pp.cm.Reds(0.5) if i == npi_season else pp.cm.Blues(c) for i,c in enumerate(np.linspace(0.5, 1, n_seasons-1))]
    
# season_start = int(np.argmax(sol.y[1] > 2*I0) * t_step / 52) + 1

# b = np.array([beta(t, p_range[i_example]) for t in sol.t])
# bmax = b[0:steps_per_season]/g

# for i in range(season_start, n_seasons):

#     lab = ''
#     if i == season_start+100:
#         lab = 'infecteds in first season'
#         # ax.text(2, 0.27, 'first season', color=colors[i-season_start])
        
#         par = patches.FancyArrowPatch(posA=(4.205, 0.262), posB=(2.7, 0.262), connectionstyle="arc3, rad=0.385", lw=0, color=colors[i-season_start], alpha=0.6,
#                                       arrowstyle=ArrowStyle('Simple', head_length=10, head_width=16, tail_width=10))
#         ax.add_patch(par)
        
#         text = CurvedText(
#             x = np.linspace(3.2, 5, 100),
#             y = 100*[0.259],
#             text='first season',
#             color='white',
#             va = 'center',
#             axes = ax,
#             fontsize='x-small'
#         )
        
#     elif i == npi_season+1:
#         lab = 'infecteds after NPI season'
#         # ax.text(3.075, 0.21, 'after NPI season', color=colors[i-season_start])
#         par = patches.FancyArrowPatch(posA=(4.205, 0.22), posB=(3.01, 0.22), connectionstyle="arc3, rad=0.31", lw=0, color=colors[i-season_start], alpha=0.6,
#                                       arrowstyle=ArrowStyle('Simple', head_length=10, head_width=16, tail_width=10))
#         ax.add_patch(par)
        
#         text = CurvedText(
#             x = np.linspace(3.2, 5, 100),
#             y = 100*[0.219],
#             text='after NPI season',
#             color='white',
#             va = 'center',
#             axes = ax,
#             fontsize='x-small'
#         )
        
#     elif i == n_seasons-1:
#         lab = 'infecteds in other seasons'
#         ax.text(4.5, 0.125, 'other seasons', color=colors[i-season_start])
        
#     S_plot = sol.y[0][i*steps_per_season:(i+1)*steps_per_season]   
#     I_plot = sol.y[1][i*steps_per_season:(i+1)*steps_per_season]
#     b_plot = b[0:steps_per_season]
    
#     reff = b_plot*S_plot/g

#     ax.plot(theta, I_plot, lw=3, color=colors[i-season_start], label=lab, alpha=1, zorder=n_seasons if i==npi_season+1 else n_seasons-i)

# ylim = 1.1*max(sol.y[1])
# ax.set_xticks(np.linspace(1/2*1/12*2*np.pi, 1/2*1/12*2*np.pi + (1-1/12)*2*np.pi, 12), calendar.month_abbr[7:] + calendar.month_abbr[1:7])
# ax.xaxis.set_tick_params(grid_alpha=0)
# ax.yaxis.set_tick_params(labelsize='small', grid_alpha=0.4)
# ax.set_ylim(0,ylim)

# theta = np.linspace(0, 2*np.pi, 13)
# ax.vlines(theta[:-1], 0, 10000, linewidth=1, color='grey', alpha=0.4)

# ax.grid(zorder=0)
# ax.spines['polar'].set_visible(False)


# ######### reproduction number #########
ax = ax3

season_start = int(np.argmax(sol.y[1] > 2*I0) * t_step / 52)

t_plot = np.linspace(0, season_length, steps_per_season)

b = np.array([beta(t, p_range[i_example]) for t in sol.t])
bmax = np.max(b)

s_colors = ['darkgreen', 'limegreen']
r_colors = ['darkred', 'red']
lst = ['--', '-']

ax32 = ax.twinx()
    
ax32.hlines(1, t_plot[0], t_plot[-1], ls='--', color=sb_grey, alpha=0.2)

for i in [npi_season+1, n_seasons-1]:

    S_plot = sol.y[0][i*steps_per_season:(i+1)*steps_per_season]    
    I_plot = sol.y[1][i*steps_per_season:(i+1)*steps_per_season]
    b_plot = b[i*steps_per_season:(i+1)*steps_per_season]
 
    ax.plot(t_plot, S_plot, ls='-', lw=2, color=sb_green, alpha=1 if i==npi_season+1 else 0.2)
    
    r0 = b_plot/g
    reff = b_plot/g*S_plot
    
    ax32.plot(t_plot, reff, ls='-', lw=2, color=sb_red, alpha=1 if i==npi_season+1 else 0.2)
    ax32.plot(t_plot, r0, lw=2, color=sb_grey, alpha=0.3, label=r'')


ax.annotate(r'$\mathbf{R_{e}}$', (20, 0.75), (12, 0.7), color=sb_red, fontweight='bold',
            arrowprops=dict(arrowstyle='-', facecolor=sb_red, edgecolor=sb_red))
ax.text(6, 0.775, r'$\mathbf{S}$', color=sb_green, fontweight='bold')
ax.text(37, 0.7, r'$\mathbf{R_{0}}$', color=sb_grey, fontweight='bold')

ax.set_xticks(np.linspace(0, 52, 13), [])
ax.set_xticks(np.linspace(2, 50, 12), minor=True)
ax.set_xticklabels(calendar.month_abbr[7:] + calendar.month_abbr[1:7], minor=True)
ax.tick_params(axis='x', which='minor', color='white')

ax.set_xlabel('time in season')
ax.set_ylabel('fraction of population')
# ax.set_ylim(0, 1.2*np.max(first_seasons + npi_seasons + other_seasons) )

ax.set_ylim(0, 1)
ax32.set_ylim(0, 1.1*bmax)

ax32.set_ylabel(r'reproduction number')

######### peakshift ########
# ax = ax4

# n_pre_npi = 1
# n_post_npi = 4

# peakmap = None

# start_season_at = 4*int(steps_per_season/12)   

# for j, sol in enumerate(sols): 
    
#     I_plot = sol.y[1][start_season_at + (npi_season-n_pre_npi)*steps_per_season:((npi_season-n_pre_npi)+n_post_npi)*steps_per_season]
    
#     if peakmap is None:
#         peakmap = I_plot
#     else:
#         peakmap = np.vstack((peakmap, I_plot))


# y = range(peakmap.shape[0]+1)
# x = range(peakmap[0].shape[0]+1)

# cvals  = [0, 1]
# colors = ["snow", sb_red]

# norm=pp.Normalize(min(cvals),max(cvals))
# tuples = list(zip(map(norm,cvals), colors))
# cmap = mcol.LinearSegmentedColormap.from_list("", tuples)

# pmesh = ax.pcolormesh(x, y, peakmap, cmap=cmap)

# from scipy.signal import argrelextrema
# peaks = argrelextrema(peakmap[0,:], np.greater)[0]

# for i in range(0, len(peaks)):
#     ax.axvline(int(peaks[i]), 0, 1, ls='--', lw=2, color=sb_grey, alpha=0.75)


# ax.set_ylabel('NPI efficacy during NPI season ($p$)')
# ax.set_yticks([i/4*peakmap.shape[0] for i in range(5)])
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=peakmap.shape[0]))

# start_month = 11
# n_months = int(t_step*len(x)/4)-2
# xtl = [calendar.month_abbr[1:][(i-1)%12][0] for i in range (start_month, start_month + n_months-1)]

# # ax.set_xlabel('\nmonth')
# # ax.text(0.48, -0.2, 'time', transform = ax.transAxes)
# ax.set_xticks(np.linspace(0, len(x), n_months), [])
# ax.set_xticks(np.linspace(0.5*len(x)/n_months, len(x)-0.5*len(x)/n_months, n_months-1), xtl, minor=True)
# ax.tick_params(axis='x', which='minor', color='white')

# ax.text(0.11, 0.8, r'normal timing ($p=0$)', color=sb_grey, transform = ax.transAxes)

# ax.annotate('', (0.59, 0.225), (0.632, 0.225), color=sb_grey, arrowprops=dict(facecolor=sb_grey, alpha=0.5, lw=0, width=10, headwidth=15), xycoords = ax.transAxes)
# # ax.text(0.41, 0.2125, 'earlier', color='white', transform = ax.transAxes, fontsize='small')

# ax.annotate('', (0.85, 0.95), (0.905, 0.95), color=sb_grey, arrowprops=dict(facecolor=sb_grey, alpha=0.5, lw=0, width=10, headwidth=15), xycoords = ax.transAxes)
# # ax.text(0.805, 0.93725, 'earlier', color='white', transform = ax.transAxes, fontsize='small')

# ax.text(0.03, -0.13, 'normal season', transform = ax.transAxes)
# rect = patches.Rectangle((0, -0.005), 0.183, -0.15, linewidth=0, edgecolor='crimson', facecolor='lightgrey', alpha=0.2, transform = ax.transAxes, clip_on=False)
# ax.add_patch(rect)

# ax.text(0.28, -0.13, 'NPI season', transform = ax.transAxes)
# rect = patches.Rectangle((0.182, -0.005), 0.274, -0.15, linewidth=0, edgecolor='crimson', facecolor='crimson', alpha=0.2, transform = ax.transAxes, clip_on=False)
# ax.add_patch(rect)

# ax.text(0.53, -0.13, 'NPI season + 1', transform = ax.transAxes)
# rect = patches.Rectangle((0.456, -0.005), 0.273, -0.15, linewidth=0, edgecolor='crimson', facecolor='lightgrey', alpha=0.2, transform = ax.transAxes, clip_on=False)
# ax.add_patch(rect)

# ax.text(0.8, -0.13, 'NPI season + 2', transform = ax.transAxes)
# rect = patches.Rectangle((0.729, -0.005), 0.273, -0.15, linewidth=0, edgecolor='crimson', facecolor='grey', alpha=0.2, transform = ax.transAxes, clip_on=False)
# ax.add_patch(rect)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# fig.colorbar(pmesh, ax=ax, shrink=0.5, fraction=0.05, pad=0.02, label='prevalence, $I$')

pp.tight_layout()
# pp.subplots_adjust(wspace=0.25, hspace=0.45)
pp.savefig('model_results.pdf', bbox_inches='tight')
