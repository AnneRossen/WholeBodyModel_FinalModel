# %%
#!/usr/bin/env python
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
from LoadParameters import *
from production_rates import *
from differential_equations import *
from plot_save_results import *
from digitalisation import *

###########################################################
###########################################################
# %%
#########################
# Organs included in simulation:
#organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose']
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Set basal values for glucose, insulin and glucagon
G0 = 5 #[mmol/L]
I0 = 15.1765 #[mU/L]
Gamma0 = 100 #[ng/L]

# Time and Food eaten at time t=0 [GLC,AA,TGL]
weight = 73 #kg
height = 176 #cm
age = 30 #years
REE  = 10*weight + 6.25*height - 5*age + 5 #male
macro_ratio = np.array([0.51,0.18,0.31]) # [%]
macro_daily_energy = macro_ratio * REE # [kcal]
macro_daily_energy /= np.array([4.15,5.65,9.4]) # Converting [kcal] to [g]
macro_daily_energy *= np.array([1/180,1/89.1,1/860])*1000  # Converting [g] to [mmol]
meal = macro_daily_energy/3


t_meals = [0, 60*5, 60*10]
#t_meals = []
t_span = [0, 60*24*2]

######################################################
# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()
t, sol, solution = Model_solver(p, x0, t_span, meal, t_meals)
######################################################

# Valueas I aim for:
print(f"Plasma glucose peak (50 min): {t[np.argmax(sol[np.where(p.organs == 'Heart')[0] * len(p.metabolites) + np.where(p.metabolites == 'GLC')[0]])]}")
print(f"Plasma triglyceride peak (240 min): {t[np.argmax(sol[np.where(p.organs == 'Heart')[0] * len(p.metabolites) + np.where(p.metabolites == 'TGL')[0]])]}")

# check with data
plot_results(t[t<20*60], sol[:,t<20*60], p,
            organ_s = ["Heart"],
            metabolite_s = ["FFA", "TGL" ,"GLC", "INS"],
            digitalisation_points = [McQuiad_NEFA, McQuiad_TGL, McQuiad_GLC, McQuiad_INS])

# Visualisation: ALL
plot_results(t, sol, p,
            organ_s = p.organs,
            metabolite_s = p.metabolites)

# %%
plot_results(t, sol, p,
            organ_s = ["Adipose upper"],
            metabolite_s = ["FFA", "TGL" ,"GLC", "INS"],
            digitalisation_points = [McQuiad_NEFA, McQuiad_TGL, McQuiad_GLC, McQuiad_INS])


###########################################################
###########################################################


# %%
# Display Results: SIMO
plot_results(t[t<5*60], sol[:,t<5*60], p,
            organ_s = ["SIMO"],
            metabolite_s = ["GLC","AA"])

plot_results(t[t<10*60], sol[:,t<10*60], p,
            organ_s = ["SIMO"],
            metabolite_s = ["TGL"])

# GLUCOSE 99% removed:
prc = (1-0.99)
t_S = float(t[ sol[-18,:] < sol[-18,1] * prc][0]/60)
t_J = float(t[ (t > t[sol[-15,:] == max(sol[-15,:])]) & (sol[-15,:] < sol[-18,1] * prc) ][0])/60
t_D = float(t[ (t > t[sol[-12,:] == max(sol[-12,:])]) & (sol[-12,:] < sol[-18,1] * prc) ][0])/60
t_L = float(t[ (t > t[sol[-9 ,:] == max(sol[-9 ,:])]) & (sol[-9 ,:] < sol[-18 ,1] * prc) ][0])/60
t_LY = float(t[ (t > t[sol[-4 ,:] == max(sol[-4 ,:])]) & (sol[-4 ,:] < sol[-16 ,1] * prc) ][0])/60

print(f"{(1-prc)*100}% removed after: \n"
      f"STOMACH (GLC): {t_S} h   --  \n" 
      f"JEJUNUM (GLC): {t_J} h   --  \n"
      f"DELAY (GLC): {t_D} h   -- {t_D-t_S} h  \n"
      f"ILEUM (GLC): {t_L} h   -- {t_L-t_D} h  \n"
      f"LYMPHATICS (TGL): {t_LY} h   -- {t_LY-t_L} h  \n")

 
# %%
# check with data
plot_results(t[t<25*60], sol[:,t<25*60], p,
            organ_s = ["Heart"],
            metabolite_s = ["FFA", "TGL", "GLC", "INS", ],
            digitalisation_points = [McQuiad_NEFA, McQuiad_TGL, McQuiad_GLC, McQuiad_INS])

# %%
plot_results(t[t<20*60], sol[:,t<20*60], p,
            organ_s = ["Liver","Heart","Muscle","Kidney"],
            metabolite_s = ["INS"])

# %%
plot_results(t[t<20*60], sol[:,t<20*60], p,
            organ_s = ["Gut"],
            metabolite_s = p.metabolites)


#############################################################
#############################################################
#############################################################

# %%
# Organs included in simulation:
organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose']
#organs = ['Brain', 'Heart', 'Gut', 'Liver', 'Kidney', 'Muscle', 'Adipose upper', 'Adipose lower']

# Set basal values for glucose, insulin and glucagon
G0 = 5 #[mmol/L]
I0 = 15.1765 #[mU/L]
Gamma0 = 100 #[ng/L]

# Time and Food eaten at time t=0 [GLC,AA,TGL]
#        kg        cm      years   
BMR  = 10*73 + 6.25*173 - 5*30 + 5
macro_ratio = np.array([0.55,0.20,0.25]) # [%]
macro_daily_energy = macro_ratio * BMR # [kcal]
macro_daily_energy /= np.array([4,4,9]) # Converting [kcal] to [g]
meal = macro_daily_energy/3
meal *= np.array([1/180,1/89.1,1/860])*1000  # Converting [g] to [mmol]

#meal = np.array([0.6,0.24,0.16])*100 #[g] (total 100g) 
#meal *= np.array([1/180,1/89.1,1/860])*1000  # Converting [g] to [mmol]

t_meals = [0]
t_span = [0, 60*5]

######################################################
# Load parameters and solve differential equations:
p = Parameters(organs)
x0 = p.get_initial_values(G0, I0, Gamma0)
p.efficiency_convertion()


# %%
def extract_hormonal_params(p):
    params = list()
    """
    # Set Insulin params
    #I_keys = ['F_(LIC)', 'F_(KIC)', 'F_(PIC)','beta_(PIR1)', 'beta_(PIR2)', 'beta_(PIR3)', 'beta_(PIR4)', 'beta_(PIR5)','M1', 'M2','alpha','beta','K', 'Q0', 'y']
    I_keys = ['F_(LIC)']
    for key in I_keys:
        params.append(p.I[key])
    
    # Set Glucagon params
    gamma_keys = ['r_(MGammaC)']
    for key in gamma_keys:
        params.append(p.gamma[key])
  
    # Set mu (reshape)
    #mu_shape = p.mu.shape
    #p.mu = np.array(param_vector[idx:idx + np.prod(mu_shape)]).reshape(mu_shape)
    """
    for Val in p.Vm["Brain"]:
        params.append(Val)
    return params


def set_hormonal_params(p, param_vector):
    
    idx = 0
    """
    # Set Insulin params
    #I_keys = ['F_(LIC)', 'F_(KIC)', 'F_(PIC)','beta_(PIR1)', 'beta_(PIR2)', 'beta_(PIR3)', 'beta_(PIR4)', 'beta_(PIR5)','M1', 'M2','alpha','beta','K', 'Q0', 'y']
    I_keys = ['F_(LIC)']
    for key in I_keys:
        p.I[key] = param_vector[idx]
        idx += 1
    
    # Set Glucagon params
    gamma_keys = ['r_(MGammaC)']
    for key in gamma_keys:
        p.gamma[key] = param_vector[idx]
        idx += 1
    """
    try:
        for i, val in enumerate(p.Vm["Brain"]):
            p.Vm["Brain"][i] = param_vector[i]
            idx += 1
    except:
        print("Error while converting parameters:")
        print("param_vector:", param_vector)
        print("Vm[Brain]:", p.Vm["Brain"])
        raise
    # Set mu (reshape)
    #mu_shape = p.mu.shape
    #p.mu = np.array(param_vector[idx:idx + np.prod(mu_shape)]).reshape(mu_shape)


def objective_comparison(param_vector, p, x0, McQuiad_GLC, McQuiad_TGL, McQuiad_INS):
    set_hormonal_params(p, param_vector)
    print(param_vector)
    t, y, solution = Model_solver(p, x0, t_span, meal, t_meals)
    
    #if not solution.success:
    #    return np.inf

    # Compare with data
    y_GLC = y[np.where(p.organs == "Heart")[0] * len(p.metabolites) + np.where(p.metabolites == "GLC")]
    y_TGL = y[np.where(p.organs == "Heart")[0] * len(p.metabolites) + np.where(p.metabolites == "TGL")]
    y_INS = y[np.where(p.organs == "Heart")[0] * len(p.metabolites) + np.where(p.metabolites == "INS")]
    
    
    interp_H_GLC = interp1d(t, y_GLC, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_H_TGL = interp1d(t, y_TGL, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_H_INS = interp1d(t, y_INS, kind='linear', bounds_error=False, fill_value='extrapolate')

    points_GLC = McQuiad_GLC[:,0] < t_span[1]/60
    points_TGL = McQuiad_TGL[:,0] < t_span[1]/60
    points_INS = McQuiad_INS[:,0] < t_span[1]/60

    residuals_GLC = McQuiad_GLC[points_GLC,1] - interp_H_GLC(McQuiad_GLC[points_GLC,0])
    #residuals_TGL = McQuiad_TGL[points_TGL,1] - interp_H_TGL(McQuiad_TGL[points_TGL,0])
    residuals_INS = McQuiad_INS[points_INS,1] - interp_H_INS(McQuiad_INS[points_INS,0])
    residuals = np.append(residuals_GLC, residuals_INS)
    
    #return residuals #minimize
    return np.sum(residuals**2) #least square

param_init = extract_hormonal_params(p)
bounds = [(param*0.99, param*1.01) for param in param_init]

res = minimize(objective_comparison,
               param_init,
               args=(p, x0, McQuiad_GLC, McQuiad_TGL, McQuiad_INS),
               method='L-BFGS-B',
               bounds=bounds,
               #options={'maxiter': 2, 'disp': True})
               options={'disp': True})

# %%
#Optimized params:
res.x





# %%
