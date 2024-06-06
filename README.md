## A machine learning model for predicting diagnostic transition to schizophrenia relies on healthcare utilization patterns
### Aparajita Kashyap, Noémie Elhadad, Steven A. Kushner, Shalmali Joshi

#### Project Overview
Schizophrenia is a highly burdensome mental health disorder; early detection of schizophrenia will enable interventions and monitoring that improve clinical outcomes. We develop a machine learning model using large-scale observational health data that predicts schizophrenia onset for psychosis patients. We further assess the robustness of our model by stratifying across race, gender, and levels of healthcare utilization. 

We use data from the Merative Marketscan Multi-state Medicaid dataset, which includes longitudinal administrative claims records for Medicaid users from 2008-2017. We use a validated phenotype to identify patients with at least 7 years of observation and at least one non-schizophrenia psychosis diagnosis prior to the index date (initial schizophrenia diagnosis or end of observation). As input into the model, we include prior conditions, medications, laboratory events, procedures, and visits. Patients are sampled at regular intervals between psychosis and the data censor date (90 days prior to initial schizophrenia diagnosis or end of observation) to enable prediction at different times in a patient’s trajectory. 

The Medicaid cohort consists of 23,781 psychosis patients of which 1,147 (4.8%) develop schizophrenia. The XGBoost model showed the highest performance: AUROC of 0.986 [95% CI: 0.985, 0.987]; sensitivity of 0.926 [0.920, 0.931]; specificity of 0.962 [0.961, 0.962]; and positive predictive value of 0.625 [0.618, 0.634]. The most important features relate to healthcare utilization (visit length of stay), routine care (oral and vision exams), and psychiatric care (antipsychotics, psychiatric diagnostic evaluation). Our model exhibits high performance and potential for clinical utility for predicting schizophrenia onset amongst patients with prior psychosis diagnoses. This model has utility both at psychosis onset and as a patient’s trajectory evolves over time.

#### Instructions
Data Processing: run CohortCreation_PatientID to identify patients who belong in the schizophrenia-positive and schizophrenia-negative cohorts. Then run temporaldata_creation to pull relevant data from the following tables: condition occurrence, visit occurrence, drug era, procedure occurrence, measurement. Deciding_VisitSplit is a tool for identifying the appropriate frequency (every nth visit) with which to sample patients, and Creation_ModelInput takes in output from temporaldata_creation.py and outputs a csv that can be used as input into the model. 

Run Models: run_model includes preprocessing code for splitting and scaling, as well as the XGBoost model ultimately used in the paper. The baseline models code allows you to create models without re-processing the data. 

Evaluation: the scripts correspond to the following figures: 
- Population Results: Table 2, eFigure 2
- Model Interpretation: Figure 1, eFigure 3
- Stability Analysis: Figure 2
- Healthcare Utilization: Figure 3
- Demographic Robustness: Table 2
- Feature Group Interpretation: eFigure 4
- Stability Analysis: 
