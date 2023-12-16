## Predicting schizophrenia onset from a cohort of psychosis patients
### Aparajita Kashyap

The goals of our work are to create a clinically actionable model that can predict schizophrenia onset from a cohort of patients with psychosis and to identify the features that drive inequity in the model. We leverage insurance claims from the IBM MarketScan Multi-state Medicaid dataset to develop an XGBoost model for schizophrenia prediction. This model is able to learn about patients at different points in their mental health trajectory and predicts schizophrenia with high fidelity (AUROC = 0.90) and specificity (0.99), but limited sensitivity (0.80). However, we find major disparities in model performance for both race (Black/White) and gender (Male/Female), and take a data-driven approach to understanding the reasons for these disparities. We find that differences in healthcare utilization and prescription medication access are potential drivers of model failure and model disparities. 

