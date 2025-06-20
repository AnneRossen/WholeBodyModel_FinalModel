# WholeBodyModel_FinalModel
This project presents an enhanced whole-body model by Carstensen et al.
The particular focus for the enhancmets are **lipid absorption**, **lipid dynamics** and **adipose tissue physiology**. 
A recreation of the Carstensen et al. can be found here: https://github.com/AnneRossen/WholeBodyModel_BaseModel.

## Key Enhancements

- **Revised Gastrointestinal (GI) Sub-model**  
  Incorporates **lymphatic fat absorption**, enabling physiologically accurate simulation of postprandial lipid transport.
- **Anatomically Distinct Adipose Compartments**  
  Separates **upper-body** and **lower-body** fat depots to capture regional differences in lipid storage and mobilization.
- **Expanded Lipid Pathway Resolution**  
  Improves representation of plasma triglyceride (TG), free fatty acid (FFA), and chylomicron kinetics across fasting and feeding states.
- **Metabolic Heterogeneity Support**  
  Enables simulation of different body compositions (e.g., top-heavy, bottom-heavy), insulin resistance, and obesity-related dysfunctions.

## Program structure

- `data/` – All parameters are saved in excel files in this data folder
- `Main.py` – Main simulation script
- `LoadParameters.py` – Script defining parameter class p
- `production_rates.py` – Script defining reaction rate function
- `differential_equations.py` – Script defining ODE system
- `plot_save_results` – Script defining plotting function
- `digitalisation.py` – Script that consist of extracted and digitalisaed data points using [atomiris](https://automeris.io/)

## How to run
1. Ensure all required Python are installed. Allapckages are listen at the top in Main.py.
2. Execute `Main.py` to run the simulation and generate output plots.

## Reference

Carstensen PE, Bendsen J, Reenberg AT, Ritschel TKS, Jørgensen JB.  
**A whole-body multi-scale mathematical model for dynamic simulation of the metabolism in man.**  
*IFAC-PapersOnLine*, 2022. [DOI: 10.1016/j.ifacol.2023.01.015](https://doi.org/10.1016/j.ifacol.2023.01.015)
