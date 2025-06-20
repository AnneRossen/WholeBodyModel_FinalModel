# %%
#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys, math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema



# %%
def plot_results(t, sol, p,
                 organ_s = ["Heart"],
                 metabolite_s = ["GLC", "AA", "GLY", "TGL", "FFA", "GLR"],
                 digitalisation_points = None
                 ):
    """
    Plot selected metabolite(s) for selected organ(s)

    Args:
        t (np.array(x,)): time
        sol (np.array(y,x)): Concentration of metabolites over time
        p (class): parameters
        organ_s (list, optional): organs to plot
        metabolite_s (list, optional): metabolites to plot
    """
        
    # Input control: strings --> list
    if isinstance(metabolite_s, str):
        metabolite_s = np.array([metabolite_s]).tolist()
    if isinstance(organ_s, str):   
        organ_s = np.array([organ_s]).tolist()
    if isinstance(metabolite_s, np.ndarray):   
        metabolite_s = metabolite_s.tolist()
    if isinstance(organ_s, np.ndarray):   
        organ_s = organ_s.tolist()
    
    # Convert time: [min] --> [h]
    if t[-1] < 60*24*2:
        t_plot = t/60
        t_unit = 'time [h]'
    else:
        t_plot = t/60/24
        t_unit = 'time [d]'
    n = len(p.metabolites) 
    n_s = len(metabolite_s)   
    
    # GI sub-model - special plot
    SIMO_organs = np.array(["Stomach", "Jejunum", "Delay", "Ileum", "Lymphatics"])    
    SIMO_metabolites = np.array(["GLC", "AA", "TGL"])
    n_simo = len(SIMO_metabolites)
    if "SIMO" in organ_s: 
        organ_s.extend(SIMO_organs)
        organ_s.remove("SIMO")
        if len(sol) != len(p.organs)*n + len(SIMO_organs)*n_simo + 3:
            print(f"Plot ERROR: number of compartments have changed and is causing plotting issues")
            print(f"{len(sol)} != {len(p.organs)}*{n} ((MAIN)) + {len(SIMO_organs)}*{n_simo} ((SIMO)) + 3 ((INS))= {len(p.organs)*n + len(SIMO_organs)*3+3})))")
    
    # Colors and theme:
    plt.style.use('bmh')
    bmh_organ = np.array(['Brain',     'Heart',     'Gut',       'Liver',     'Kidney',    'Muscle',     'Adipose',   'Adipose upper',    'Adipose lower',   'Stomach',   'Jejunum',   'Delay',    'Ileum',      'Lymphatics'])
    bmh_color =          ['#5c4934', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#348ABD', '#0072B2',        '#56B4E9',      '#4e0550', '#b01212', '#ca6b02', "#A79B00", '#009E73','#580F41','#653700','#F0E442']
    metabolite_unit = {m: 'mU/L' if m in ["INS", "GLU"] else 'mM/L' for m in p.metabolites}
    
    # Figure size og subplot composition:
    cols = 4 if n_s>9 or n_s==8 else 3 if n_s > 4 else 2 if n_s > 1 else 1
    rows = math.ceil(n_s / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.5*cols, 2*rows))
    axes = np.array(axes).flatten()  # Flatten in case of 1 row
    
    # Sub plot for each metabolite
    for i, metabolite in enumerate(metabolite_s):
        #print(metabolite)
        # Graph for each organ  
        for organ in organ_s:
            if metabolite == "PRO" and organ not in ["Muscle"]:
                continue
            if metabolite == "TGL_AP" and organ not in ['Adipose', 'Adipose upper', 'Adipose lower']:
                continue
            if metabolite == "GLY" and organ not in ['Liver', 'Muscle']:
                continue
            if organ in SIMO_organs: 
                metabolite_idx = np.where(SIMO_metabolites == metabolite)[0]
                organ_idx = int(np.where(SIMO_organs == organ)[0])
                idx = len(p.organs)*n + organ_idx*n_simo + metabolite_idx
            else:
                metabolite_idx = np.where(p.metabolites == metabolite)[0]
                organ_idx = int(np.where(p.organs == organ)[0])
                idx = organ_idx*n + metabolite_idx
            # only GLC, AA and TGL is tracked in SIMO organ compartments:
            if (organ in SIMO_organs and metabolite not in SIMO_metabolites):
                pass
            else:
                organ_color = bmh_color[int(np.where(bmh_organ == organ)[0])]
                axes[i].plot(t_plot, sol[idx,:][0], label=organ, color=organ_color, linewidth=1, alpha=0.8)
        
        # Labels and axis
        axes[i].set_xlabel(t_unit, fontsize=8)
        axes[i].set_ylabel(f'{metabolite} Concentration [{metabolite_unit[metabolite]}]', fontsize=8)
        axes[i].grid(True, alpha=0.7)
        axes[i].tick_params(axis='x', labelsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        
        # local minimums and maximums: GLC
        """
        try:
            idx_H_GLC = np.where(p.organs == "Heart")[0]*n + np.where(p.metabolites == "GLC")[0]
            local_maxs = argrelextrema(sol[idx_H_GLC, :][0], np.greater, order=2)[0].flatten()
            local_mins = [argrelextrema(sol[idx_H_GLC, :][0], np.less, order=2)[0][1]]
            for extrema in np.concatenate([local_maxs, local_mins]):
                axes[i].axvline(x=t_plot[extrema], color='black', linestyle='--', linewidth=1, alpha=0.5)
        except:
            pass
        """    
        # Day indicator(s):
        """
        try:
            t_days = np.arange(0, max(t_plot), 24)[1:]
            for days in t_days:
                axes[i].axvline(x=days, color='black', linestyle='-', linewidth=1, alpha=0.2)
        except:
            pass
        """
        # Optional plot validation points using digitalisation:
        if digitalisation_points:
            try:
                x_values, y_values = zip(*[(x, y) for x, y in digitalisation_points[i] if x < t_plot[-1]])
                if t[-1] > 60*24*2:
                    x_values = x_values / 24 #days
                #axes[i].scatter(x_values, y_values, color='red', label='XX', marker='.', s=10) 
                axes[i].plot(x_values, y_values, label='XX', color='red', linestyle='dotted' , linewidth=1, alpha=0.5, marker='.', markersize=4)
            except: 
                pass
    
    # Legend
    custom_legend = []
    for organ in organ_s:
        organ_color = bmh_color[int(np.where(bmh_organ == organ)[0])]
        custom_legend.append(Line2D([0], [0], color=organ_color, linestyle='-'))
    if digitalisation_points:
        custom_legend.append(Line2D([0], [0], color='red', linestyle='dotted', linewidth=1,  alpha=0.5, marker='.', markersize=4))
        organ_s.append("Validation points")
        
    
    fig.legend(custom_legend, organ_s, loc='lower center', ncol = len(organ_s), fontsize=8)
    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    
    # Plot
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


"""
plot_results(t[t<20*60], sol[:,t<20*60], p,
            organ_s = ["SIMO", "Heart"],
            metabolite_s = ["GLC", "AA", "TGL", "INS"])
"""

# %%
