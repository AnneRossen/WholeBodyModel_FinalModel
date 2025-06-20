#!/usr/bin/env python
# %%
import numpy as np
import sys, math
from joblib import Parallel, delayed

# %%
def get_r(p, organ):
    """
    Calulates reaction rate r(C) using Michaelis Menten kinetics
    """
    C = p.C[organ]
    Vm = p.Vm[organ]
    Km = p.Km[organ]
    rates = np.zeros(p.S.shape[0])
    
    # Reduce to reactions that occour
    active = Vm!=0
    Vm = Vm[active]
    Km = Km[active]
    rates_a = np.zeros_like(Vm)
    S_active = p.S[active,:]
    
    # Standard form Michaelis-Menten (V*[A] / K1+[A])
    MM1 = (S_active < 0).sum(axis=1) == 1
    metabolites = np.argmax(S_active[MM1] < 0, axis=1)
    C1 = C[metabolites]
    C1[C1<0] = 0
    numerator = Vm[MM1] * C1
    denominator = Km[MM1][:,0] + C1
    rates_a[MM1] = numerator/denominator
    
    # Two substrate Michaelis-Menten (V*[A]*[B]) / (K1 + K2*[A] + K3*[B] + [A]*[B]) 
    MM2 = (S_active < 0).sum(axis=1) > 1
    metabolites = np.argwhere(S_active[MM2] < 0)[:,1].reshape(-1,2)
    C2 = C[metabolites]
    C2[C2<0] = 0
    numerator = Vm[MM2] * C2[:,0] * C2[:,1]
    denominator = Km[MM2][:,0] + Km[MM2][:,1]*C2[:,0] + Km[MM2][:,2]*C2[:,1] + C2[:,0]*C2[:,1]
    rates_a[MM2] = numerator/denominator
    rates[active] = rates_a
    
    return rates

def process_r(p, organ):
    r_val = get_r(p, organ)
    I_val = p.C[organ][-2]
    Gamma_val = p.C[organ][-1]
    return organ, r_val, I_val, Gamma_val


def ProductionRates(p, J, L, II, QQ):
    """
    quick_r = Parallel(n_jobs=-1)(
        delayed(process_r)(p, organ) for organ in p.organs)
    
    for organ, r_val, I_val, Gamma_val in quick_r:
        # Michaelis-Menten base reaction rates
        p.r[organ] = r_val
        
        # Extract Insulin and Glucagon concentrations.
        p.I[organ] = I_val
        p.Gamma[organ] = Gamma_val
    
    """
    for organ in p.organs:
        # Michaelis-Menten base reaction rates
        p.r[organ] = get_r(p, organ)
        
        # Extract Insulin and Glucagon concentrations.
        p.I[organ] = p.C[organ][-2]
        p.Gamma[organ] = p.C[organ][-1]
    
    ##################################################

    # SIMO-submodel
    [p.GI["GLC"], p.GI["AA"], p.GI["TGL"]] = p.GI["k_(gj)"]*J + p.GI["k_(gl)"]*L
    
    ##################################################
    
    # Insulin submodel
    p.I["X"] = (18 * p.C["Heart"][0])**p.I["beta_(PIR1)"] / (p.I["beta_(PIR2)"]**p.I["beta_(PIR1)"] + p.I["beta_(PIR3)"] * (18*p.C["Heart"][0])**p.I["beta_(PIR4)"])
    p.I["Y"] = p.I["X"]**p.I["beta_(PIR5)"]
    p.I["P_inf"] = p.I["Y"]
    p.I["S"] = (p.I["M1"]*p.I["Y"] + p.I["M2"]*min(0, p.I["X"]-II)) * QQ # insuling release
    p.I["r_PIR"] = p.I["S"] / p.I["S_B"] * p.I["r_B_PIR"] # Insulin production
    
    # Insulin clearance
    p.I["r_(LIC)"] = p.I["F_(LIC)"] * (p.Q["Q_A"]*p.I["Heart"] + p.Q["Gut"]*p.I["Gut"] + p.I["r_PIR"])
    p.I["r_(KIC)"] = p.I["F_(KIC)"] * p.Q["Kidney"] * p.I["Heart"]
    p.I["r_(MIC)"] = p.I["F_(PIC)"] * p.Q["Muscle"] * p.I["Heart"]
    #p.I["r_(AIC)"] = p.I["F_(PIC)"] * (sum([p.Q[organ] for organ in p.organs if "Adipose" in organ])) * p.I["Heart"]

    # Glucagon submodel
    p.gamma["r_(PGammaC)"] = p.gamma["r_(MGammaC)"] * p.Gamma["Heart"]
    p.gamma["M_G_(PGammaR)"] = 2.93 - 2.10 * math.tanh(4.18 * (p.C["Heart"][0] / p.gamma["G_B_H"] - 0.61))
    p.gamma["M_I_(PGammaR)"] = 1.31 - 0.61 * math.tanh(1.06 * (p.I["Heart"] / p.gamma["I_B_H"] - 0.47))
    p.gamma["r_(PgammaR)"] = p.gamma["M_G_(PGammaR)"] * p.gamma["M_I_(PGammaR)"] * p.gamma["r_(B_PGammaR)"]

    
    ##################################################
    
    # Liver Hormonal influence 
    INS_activate_L = p.I["Liver"]/p.I["L_SS"]
    INS_inhibit_L  = p.I["L_SS"]/p.I["Liver"]
    GLU_activate_L = p.Gamma["Liver"]/p.gamma["L_SS"]
    GLU_inhibit_L = p.gamma["L_SS"]/p.Gamma["Liver"]
    p.r["Liver"][0]  *= (INS_activate_L) **p.mu["Liver"][0]
    p.r["Liver"][2]  *= (INS_activate_L * GLU_inhibit_L) **p.mu["Liver"][2]
    p.r["Liver"][3]  *= (INS_inhibit_L * GLU_activate_L) **p.mu["Liver"][3]
    p.r["Liver"][4]  *= (INS_activate_L * GLU_inhibit_L) **p.mu["Liver"][4]
    p.r["Liver"][5]  *= (INS_inhibit_L * GLU_activate_L) **p.mu["Liver"][5]
    p.r["Liver"][12] *= (INS_activate_L) **p.mu["Liver"][12]
    p.r["Liver"][22] *= (INS_activate_L) **p.mu["Liver"][22]
    p.r["Liver"][29]  = (p.I["r_PIR"] - p.I["r_(LIC)"]) /p.V["Liver"] # INS
    p.r["Liver"][30]  = (p.gamma["r_(PgammaR)"] - p.gamma["r_(PGammaC)"]) /p.V["Liver"] # GLUCOGEN
    
    # Kidney Hormonal influence
    p.r["Kidney"][29] = -p.I["r_(KIC)"]/p.V["Kidney"] # INS
    
    # Muscle Hormonal influence
    INS_activate_M = p.I["Muscle"]/p.I["MP_SS"]
    INS_inhibit_M  = p.I["MP_SS"]/p.I["Muscle"]
    p.r["Muscle"][0]  *= (INS_activate_M) **p.mu["Muscle"][0]
    p.r["Muscle"][2]  *= (INS_activate_M) **p.mu["Muscle"][2]
    p.r["Muscle"][4]  *= (INS_activate_M) **p.mu["Muscle"][4]
    p.r["Muscle"][5]  *= (INS_inhibit_M)  **p.mu["Muscle"][5]
    p.r["Muscle"][12] *= (INS_activate_M) **p.mu["Muscle"][12]
    p.r["Muscle"][25] *= (INS_activate_M) **p.mu["Muscle"][25]
    p.r["Muscle"][26] *= (INS_inhibit_M)  **p.mu["Muscle"][26]
    p.r["Muscle"][29]  = -p.I["r_(MIC)"]/p.V["Muscle"] # INS
    
    # Adipose Hormonal influence
    Adipose_organs = [o for o in p.organs if "Adipose" in o]
    for a in Adipose_organs:
        INS_activate_A = p.I[a]/p.I["AP_SS"]
        INS_inhibit_A  = p.I["AP_SS"]/p.I[a]
        GLU_activate_A = p.Gamma[a]/p.gamma["AP_SS"]
        
        p.I["r_(AIC)"] = p.I["F_(PIC)"] * p.Q[a] * p.I["Heart"]
        
        p.r[a][0]  *= (INS_activate_A) **p.mu[a][0]
        p.r[a][20] *= (INS_inhibit_A) **p.mu[a][20]
        p.r[a][27] *= (INS_activate_A) **p.mu[a][27]
        p.r[a][28] *= (INS_inhibit_A * GLU_activate_A) **p.mu[a][28]
        p.r[a][29]  = -p.I["r_(AIC)"]/p.V[a] # INS
    
    ##################################################
    
    # Production rate vectors (S*r)
    for organ in p.organs:
        p.R[organ] = p.S.T @ p.r[organ]
                
        #neg_metabolite_idx = np.where(p.S.T @ p.r[organ] + p.C[organ] < 0 )[0]
        #reaktion_idx = np.where(p.S[:, neg_metabolite_idx] != 0)[0]
        #p.R[organ][reaktion_idx] = [0]*len(reaktion_idx)
        """
        negatives[2] = True
        
        # No negative concentrations cap
        negatives = p.S.T @ p.r[organ] + p.C[organ] < 0 
        if any(negatives):
            print(organ)
            print(p.R[organ])
            print(p.C[organ])
            print(negatives)
            sys.exit()
            
        #p.R[organ][negatives] = 0
        """
        """
        # Error handling
        if np.any(np.isnan(p.r[organ])) or np.any(np.isinf(p.r[organ])):
            print("Warning! NaN or Inf detected in r for organ: ", organ)
            print(organ)
            print(p.r[organ])
            return np.inf
        """
    return p

# %%
