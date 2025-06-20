
# #!/usr/bin/env python
# %%
from LoadParameters import *
from production_rates import *
from scipy.integrate import solve_ivp
import numpy as np

# %%
def Model(t, X, p):    
    """
    Whole-body model differential equation system
    
    Args:
        p (class): parameters
    """    
    # Main Equations: circulating metabolites
    idx_X = 0
    n_metabolites = len(p.metabolites)
    for organ in p.organs:
        p.C[organ] = X[idx_X : (idx_X+n_metabolites)]
        idx_X += n_metabolites
        
    # Simo submodel Equations 
    S = X[-18:-15]  # Stomach 
    J = X[-15:-12] # Jejunum 
    R = X[-12:-9] # Jejunum 
    L = X[-9:-6]   # Ileum
    Ly = X[-6:-3]   # Lymphatics
    
    # Insulin release Equations
    P = X[-3]
    II = X[-2]
    QQ = X[-1]
    
    ##################################################
    
    # Calculate production rates vectors and glucagon/insulin submodel
    p = ProductionRates(p, J, L, II, QQ)
    
    # SIMO-vectors for macronutrient uptake
    GLC_idx = np.where(p.metabolites == "GLC")[0]
    AA_idx  = np.where(p.metabolites == "AA")[0]
    TGL_idx = np.where(p.metabolites == "TGL")[0]
    
    SIMO = {
        "GI": np.zeros(n_metabolites),
        "MP": np.zeros(n_metabolites),
        "AP": np.zeros(n_metabolites),
        "H":  np.zeros(n_metabolites),
    }
    
    SIMO["GI"][GLC_idx] = p.GI["GLC"]
    SIMO["GI"][AA_idx] = p.GI["AA"]
    SIMO["H"][TGL_idx] = p.GI["k_(hly)"]*Ly[2]
    
    ##################################################
    
    # Main equations
    dCdt = {
        "Brain":  (p.m * p.Q["Brain"]   @ (p.C["Heart"] - p.C["Brain"])) / p.V["Brain"] + p.R["Brain"],
        "Heart":  (p.m @ (sum([p.Q[o]*p.C[o] for o in p.organs if o not in ["Heart", "Gut"]]) - p.Q["Heart"] * p.C["Heart"]) + SIMO["H"]) / p.V["Heart"] + p.R["Heart"],
        "Gut":    (p.m * p.Q["Gut"]     @ (p.C["Heart"] - p.C["Gut"]) + SIMO["GI"]) / p.V["Gut"] + p.R["Gut"],
        "Liver":  (p.m @ (p.Q["Q_A"] * p.C["Heart"] + p.Q["Gut"] * p.C["Gut"] - p.Q["Liver"] * p.C["Liver"])) / p.V["Liver"] + p.R["Liver"],
        "Kidney": (p.m * p.Q["Kidney"]  @ (p.C["Heart"] - p.C["Kidney"])) / p.V["Kidney"] + p.R["Kidney"],
        "Muscle": (p.m * p.Q["Muscle"]  @ (p.C["Heart"] - p.C["Muscle"])) / p.V["Muscle"] + p.R["Muscle"],
    }
    # Several options for Adipose tissues
    Adipose_organs = [o for o in p.organs if "Adipose" in o]
    for organ in Adipose_organs:
        dCdt[organ] = (p.m * p.Q[organ] @ (p.C["Heart"] - p.C[organ])) / p.V[organ] + p.R[organ]/len(Adipose_organs)
    # Combine concentration derivatives
    MainEq = np.hstack([dCdt[o] for o in dCdt.keys()])


    # SIMO model
    dGIdt = {
        "S": -p.GI["k_(js)"]*S,
        "J":  p.GI["k_(js)"]*S - p.GI["k_(rj)"]*J - p.GI["k_(gj)"]*J,
        "R":  p.GI["k_(rj)"]*J - p.GI["k_(lr)"]*R,
        "L":  p.GI["k_(lr)"]*R - p.GI["k_(gl)"]*L,
        "Ly": [0,0,p.GI["TGL"]]- p.GI["k_(hly)"]*Ly
    }
    SimoEq = np.hstack([dGIdt[o] for o in dGIdt.keys()])
    
    
    # Insulin submodel
    dINSdt = {
        "P":  p.I["alpha"] * (p.I["P_inf"]-P),
        "II": p.I["beta"] * (p.I["X"]-II),
        "QQ": p.I["K"] * (p.I["Q0"]-QQ) + p.I["y"]*P - p.I["S"]
    }
    InsEq = np.hstack([dINSdt[o] for o in dINSdt.keys()])
    
    dzdt = np.concatenate([MainEq, SimoEq, InsEq])
    
    """
    # Error handling
    print(f"t={t}")
    if np.any(np.isnan(dzdt)) or np.any(np.isinf(dzdt)):
        print("Warning! NaN or Inf detected in dzdt at t =", t)
        print("dzdt shape:", dzdt.shape, " Min/Max:", np.min(dzdt), np.max(dzdt))
        sys.exit()
    """
    return dzdt

# Solve differential equations
def Model_solver(p, x0, t_span, meal, t_meals):
    
    t_start, t_end = t_span
    t_meals = sorted(set([t_meal for t_meal in t_meals if t_start <= t_meal <= t_end]))
    time_points = np.unique([t_start] + t_meals + [t_end])#[:-1]
      
    try:
        if t_meals[0] == t_start: #(onset meal)
            x0[-18:-15] += meal
    except:
        pass
    sol = np.array(x0).reshape(-1, 1)
    t = [t_start]
    x = x0
    
    # solve differential eqution in intervals around meals
    for i in range(len(time_points)-1):
        t_interval = [time_points[i], time_points[i+1]]
        print(t_interval)
        
        # Solve model
        solution = solve_ivp(Model, t_interval, x, args=(p,), method='BDF'
                            #bounds=(np.zeros(n), np.full(n, np.inf))
                            #t_eval=t_eval, rtol=1e-3, atol=1e-6
                            )
        
        # Save and initialize
        t = np.concatenate((t, solution.t.flatten()[1:]))
        sol = np.concatenate((sol, solution.y[:,1:]), axis=1)
        x = solution.y[:,-1]
        
        # Add next meal
        x[-18:-15] += meal
        
    return t, sol, solution


# %%
