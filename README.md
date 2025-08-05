# MtDNA Replication Dynamics

This repository contains all code needed for the stochastic modelling, approximate Bayesian computation (ABC), causal analysis, and all other miscellaneous mathematical and statistical analyses involved in 'Single nucleoid cellular imaging reveals the mechanics of copy number control', by Dane, Mjeku, et al.

This code:
- Builds various stochastic models, using the Gillespie algorithm, to simulate replication and degradation of mitochondrial DNA (mtDNA), fits these models to observed data using ABC, selects between these models using ABC, validates the chosen model, and plots various biological predictions outputted from the best fitted model.
- Conducts a causal analysis on the relationship between nucleoid number, mitochondrial volume, and cellular volume, using three forms of conditional independence tests, linear Granger causality, and neural Granger causality.
- Selects between different mtDNA birth rate mechanisms by careful inspection of the stochasticity present in each
- Provides code for other miscellaneous analyses including, but not limited to, Gaussian mixture modelling of mtEdU intensities, deterministic model fitting to cells exposed to ethidium bromide, deterministic model fitting to primary mouse macrophage data, and error quantification.

# Stochastic Models

We recommend starting with stochastic_models_demo.ipynb, which gives an overview of what our various stochastic models output, and how this output changes based on the parameterisation given.

Source code for the stochastic models are given in data_preprocessing/stochastic_systems.py

# Approximate Bayesian Computation

posteriors.ipynb displays ABC priors and posteriors for all models trained, and derived_posteriors.ipynb displays posteriors of quantities derived from the posterior parameters, for example the turnover rate and subpopulation proportions. If you wish to rerun the ABC from scratch, see recreate_ABC_simulations.ipynb. ABC_model_selection.ipynb runs the model selection algorithm which selects the three population model. ABC_validation.ipynb validates the selected three population model, which was only trained on the first two assays, on the third assay.

Source code for much of the ABC analysis are given in data_preprocessing/ABC_priors_and_functions.py

# Posterior Predictive Plots

bulk_posterior_predictive_plots.ipynb simulates and plots mean posterior predictive trajectories from our already saved ABC posteriors. single_cell_posterior_predictive_plots.ipynb simulates and plots posterior predictives for each individual single cell.

Source code for much of the ABC analysis are given in data_preprocessing/ABC_priors_and_functions.py

# Birth Rate Selection

Birth rate selection is found in birth_rate_selection.ipynb

# Causal Analysis

Conditional independence testing for all cell types can be found in conditional_independence_testing_CMI.ipynb and conditional_independence_testing_GCM_KCIT.rmd. Linear Granger causality testing is found in linear_granger_causality.ipynb, and neural_granger_causality.ipynb. 

Neural Granger causality functions are found in data_preprocessing/neural_granger_causality_functions.py

# Miscellaneous Analyses

Gaussian mixture modelling of mtEdU intensities, along with BIC scores, is found in guassian_mixture_figures.ipynb. Model fits to ethidium bromide exposed cells are found in degradation_data_modelling.ipynb. Model fits to primary mouse macrophage data are found in mouse_modelling.ipynb. Data for ABC and all other analysis are extracted and preprocessed in data_preprocessing/data_preprocessing.py. All data used for fitting, along with simulated ABC or posterior predictive data, can be found within the subfolders of data_preprocessing
