#!/usr/bin/env python
# %%
# Import packages
import pandas as pd
import numpy as np
import math, json, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import least_squares
# %%
# Import functions
from LoadParameters import *
from production_rates import *
from differential_equations import *
from plot_save_results import *
from digitalisation import *

# %%
######################################################
############# Common initial conditions ##############

# Basal values for glucose, insulin and glucagon
G0 = 5 # [mmol/L]
I0 = 15.1765 # [mU/L]
Gamma0 = 100 # [ng/L]

# Meal composition [GLC,AA,TGL]
weight = 73 # kg
height = 176 # cm
age = 30 # years
REE  = 10*weight + 6.25*height - 5*age + 5 #male
macro_ratio = np.array([0.51,0.18,0.31]) # [%]
macro_daily_energy = macro_ratio * REE # [kcal]
macro_daily_energy /= np.array([4.15,5.65,9.4]) # Converting [kcal] to [g]
macro_daily_energy *= np.array([1/180,1/89.1,1/860])*1000  # Converting [g] to [mmol]
meal = macro_daily_energy/3

# %%
######################################################
#################### GI submodel #####################
print("INCOMMING: GI submodel enhancement")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose']

# Time Details
t_meals = [0]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results: GI submodel
plot_results(t[t<10*60], sol[:,t<10*60], p,
            organ_s = ["SIMO"],
            metabolite_s = ["GLC","AA", "TGL"])

# time statistics:
prc = (1-0.99)
t_S = float(t[ sol[-18,:] < sol[-18,1] * prc][0]/60)
t_J = float(t[ (t > t[sol[-15,:] == max(sol[-15,:])]) & (sol[-15,:] < sol[-18,1] * prc) ][0])/60
t_D = float(t[ (t > t[sol[-12,:] == max(sol[-12,:])]) & (sol[-12,:] < sol[-18,1] * prc) ][0])/60
t_L = float(t[ (t > t[sol[-9 ,:] == max(sol[-9 ,:])]) & (sol[-9 ,:] < sol[-18 ,1] * prc) ][0])/60
t_LY = float(t[ (t > t[sol[-4 ,:] == max(sol[-4 ,:])]) & (sol[-4 ,:] < sol[-16 ,1] * prc) ][0])/60
print(f"{(1-prc)*100}% removed after: \n"
      f"STOMACH (GLC): \t\t{t_S} h\t (Peak {t[sol[-18,:] == max(sol[-18,:])]/60} h) \n" 
      f"JEJUNUM (GLC): \t\t{t_J} h\t (Peak {t[sol[-15,:] == max(sol[-15,:])]/60} h) \n" 
      f"DELAY (GLC): \t\t{t_D} h\t (Peak {t[sol[-12,:] == max(sol[-12,:])]/60} h) \n" 
      f"ILEUM (GLC): \t\t{t_L} h\t (Peak {t[sol[-9,:] == max(sol[-9,:])]/60} h) \n" 
      f"LYMPHATICS (TGL): \t{t_LY} h\t (Peak {t[sol[-4,:] == max(sol[-4,:])]/60} h) \n" )

# Display Results: Plasma dynamics
plot_results(t[t<10*60], sol[:,t<10*60], p,
            organ_s = ["Heart"],
            metabolite_s = ["GLC","AA", "TGL"])

# time statistics:
Heart_GLC_idx = int(int(np.where(p.organs == "Heart")[0])*len(p.metabolites) + np.where(p.metabolites == "GLC")[0])
Heart_AA_idx = int(int(np.where(p.organs == "Heart")[0])*len(p.metabolites) + np.where(p.metabolites == "AA")[0])
Heart_TGL_idx = int(int(np.where(p.organs == "Heart")[0])*len(p.metabolites) + np.where(p.metabolites == "TGL")[0])
print(f"Plasma times: \n"
      f"GLC:\t peak: {t[sol[Heart_GLC_idx,:] == max(sol[Heart_GLC_idx,:])]/60} h) valley: {t[sol[Heart_GLC_idx,:] == min(sol[Heart_GLC_idx,:])]/60} h)\n"
      f"AA:\t peak: {t[sol[Heart_AA_idx,:] == max(sol[Heart_AA_idx,:])]/60} h) valley: {t[sol[Heart_AA_idx,:] == min(sol[Heart_AA_idx,:])]/60} h)\n"
      f"TGL:\t peak: {t[sol[Heart_TGL_idx,:] == max(sol[Heart_TGL_idx,:])]/60} h) valley: {t[sol[Heart_TGL_idx,:] == min(sol[Heart_TGL_idx,:])]/60} h)\n" 
      )


# %%
######################################################
######## Adipose implementation check tissues ########
print("INCOMMING: Adipose compartment enhancement - 1 compartment")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results: Adipose
plot_results(t, sol, p,
            organ_s = ["Adipose"],
            metabolite_s = p.metabolites)

print("INCOMMING: Adipose compartment enhancement - 2 equal compartments")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
p.V["Adipose upper"] = p.V["Adipose"] * 0.5 # Equal subdivision of Volume
p.V["Adipose lower"] = p.V["Adipose"] * 0.5 # Equal subdivision of Volume
p.Q["Adipose upper"] = p.Q["Adipose"] * 0.5 # Equal subdivision of Flow
p.Q["Adipose lower"] = p.Q["Adipose"] * 0.5 # Equal subdivision of Flow
p.Vm["Adipose upper"] = p.Vm["Adipose"] # Equal Reaction kinetics
p.Vm["Adipose lower"] = p.Vm["Adipose"] # Equal Reaction kinetics
p.mu["Adipose upper"] = p.mu["Adipose"] # Equal Reaction kinetics
p.mu["Adipose lower"] = p.mu["Adipose"] # Equal Reaction kinetics
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results: Adipose
plot_results(t, sol, p,
            organ_s = ['Adipose upper', 'Adipose lower'],
            metabolite_s = p.metabolites)


# %%
######################################################
############### Parameter estimation #################
print("INCOMMING: Parameter estimation - comparison with data")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*20]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = ["Heart"],
            metabolite_s = ["FFA", "TGL" ,"GLC", "INS"],
            digitalisation_points = [McQuiad_NEFA, McQuiad_TGL, McQuiad_GLC, McQuiad_INS])

#%%
######################################################
################ Adipose distribtion #################
print("INCOMMING: Average Adipose distribution")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Time Details
t_meals = [60*(0+0),  60*(0+5),  60*(0+10)]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)

######################################################
print("INCOMMING: Increased Upper Adipose distribution")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
p.V['Adipose upper'] = p.V['Adipose'] * 0.8  # Edit adipose distribtion (top heavy)
p.V['Adipose lower'] = p.V['Adipose'] * 0.2  # Edit adipose distribtion (top heavy)
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)

######################################################
print("INCOMMING: Increased Lower Adipose distribution")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
p.V['Adipose upper'] = p.V['Adipose']* 0.4 # Edit adipose distribtion (bottom heavy)
p.V['Adipose lower'] * p.V['Adipose']* 0.6 # Edit adipose distribtion (bottom heavy)
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)

#%%
######################################################
################ Insulin resistance ##################
print("INCOMMING: Insulin resistance scenario")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
p.mu['Muscle'] *= 0.05        # Insulin resistance
p.mu['Adipose upper'] *= 0.05 # Insulin resistance
p.mu['Adipose lower'] *= 0.05 # Insulin resistance
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)


# %%
######################################################
###################### Obesity #######################
print("INCOMMING: Obesity scenario")

# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Time Details
t_meals = [0, 60*5, 60*10]
t_span = [0, 60*24]

# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
p.V['Adipose upper'] = p.V['Adipose']*2 * 0.60 # increased adipose volume
p.V['Adipose lower'] = p.V['Adipose']*2 * 0.40 # increased adipose volume
p.Q['Adipose upper'] *= 0.7 # Decreased adipose blood flow
p.Q['Adipose lower'] *= 0.7 # Decreased adipose blood flow
p.Q['Heart'] = p.Q['Brain'] + p.Q['Liver'] + p.Q['Kidney'] + p.Q['Muscle'] + p.Q['Adipose upper'] + p.Q['Adipose lower'] 
p.Vm['Adipose upper'][28] *= 0.5 # Decreased fat release
p.Vm['Adipose lower'][28] *= 0.5 # Decreased fat release
p.Vm['Adipose upper'][27] *= 0.50 # Decreased fat uptake
p.Vm['Adipose lower'][27] *= 0.50#0.75 # Decreased fat uptake
#p.mu['Muscle'] *= 0.5        # Insulin resistance
#p.mu['Adipose upper'] *= 0.5 # Insulin resistance
#p.mu['Adipose lower'] *= 0.5 # Insulin resistance
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)

# Display Results:
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)

# %%
