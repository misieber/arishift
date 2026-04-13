import numpy as np
from math import cos
from scipy.integrate import solve_ivp

duration_of_infectivity = 1  # week
duration_of_immunity    = 40 # weeks

beta_max = 3
beta_min = 0.6

n_seasons = 8
season_length = 52 # weeks
npi_season = 3

I0 = 10**-5
S0 = 1 - I0
y0 = np.array([S0, I0, 0])

g = 1/duration_of_infectivity
w = 1/duration_of_immunity

t_end = n_seasons*season_length
t_step = 0.1
t_steps = np.arange(0, t_end, t_step)
steps_per_season = int(season_length/t_step)


def beta(t, p):
    npi_e = 1
    if t > npi_season*season_length and t < (npi_season+1)*season_length:
        npi_e = (1-p)
    
    v = npi_e*0.5*(1 - cos((2*np.pi/season_length)*t))*(beta_max-beta_min) + beta_min

    return v


def model(t, y, p):
    S, I, R = y

    dSdt = -beta(t, p)*S*I + w*R
    dIdt = beta(t, p)*S*I - g*I
    dRdt = g*I - w*R

    return [dSdt, dIdt, dRdt]

def solver(parameters):
    return solve_ivp(model, [t_steps[0], t_steps[-1]], y0, args=[parameters], t_eval=t_steps, rtol=10**-13, atol=10**-16, method='DOP853')
