## A machine learning model for predicting diagnostic transition to schizophrenia relies on healthcare utilization patterns
### Aparajita Kashyap, Noémie Elhadad, Steven A. Kushner, Shalmali Joshi

#### Project Overview
Schizophrenia is a debilitating psychiatric disorder in which earlier clinical recognition would enhance monitoring and treatment planning. We develop a deep learning model using Medicaid administrative claims to predict the risk of schizophrenia following antecedent episodes of psychosis. We included 158,704 individuals with a history of psychosis, of whom 13,437 (8.5%) were subsequently diagnosed with schizophrenia. We used a validated phenotype to predict the individual-level likelihood of future incipient schizophrenia by leveraging temporal patterns of clinical features. External validity of the best performing model, a transformer-based neural network, was assessed using an independent, commercially insured cohort. The transformer model exhibited an AUROC of 0.791 [95% CI: 0.788, 0.794] and transported well to the external dataset (AUROC: 0.763 [0.751, 0.772]). Features with the highest information content were predominantly related to psychiatric care and healthcare utilization. Overall, we find promising performance for schizophrenia risk stratification among individuals with a history of psychosis. 

#### Running code from this repository
1. Data Processing
- *CohortCreation_PatientID:* identify patients who belong in the schizophrenia-positive and schizophrenia-negative cohorts
- *temporaldata_creation:* pull relevant data from the following tables: condition occurrence, visit occurrence, drug era, procedure occurrence, measurement
- *create_flat_df_snomed and create_grud_timedeltas:* generates flat files of the data needed for creating the dataloader objects (grud_timedeltas is needed for the GRU-D model only)
- *create_dataloaders, create_grud_dataloaders:* takes in flat files and generates dataloaders for the model
2. Run Models
- Each "run" file trains models according to the specified architecture
3. Evaluation: scripts to calculate performance for different subgroups
- *Evaluation/AFO* contains scripts to run feature importance
