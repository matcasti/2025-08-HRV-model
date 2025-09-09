
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(ggplot2)
library(rstan)

## Load data and parameters
params <- readRDS(file = "data/simulation_parameters.RDS")
freq_bands <- readRDS(file = "data/freq_bands.RDS")
simulated_data <- readRDS(file = "data/simulated_data.RDS")

# -------------------------------------------------------------------------

## Compile the stan model
model <- rstan::stan_model(file = "models/rri_model.stan")

for(i in 1:3) {
  model_fit <- rstan::sampling(
    object = model,
    pars = c(
      "lambda_log","phi_log","tau_logit","delta_logit",
      "alpha_r_logit","beta_r_logit","c_r_logit",
      "alpha_s_logit","beta_s_logit","c_s_logit",
      "c_c_logit", "b_log", "w_logit", "sigma_u",
      "y_base_log", "y_pert_log",
      "lambda","phi","tau","delta",
      "alpha_r","beta_r","c_r",
      "alpha_s","beta_s","c_s",
      "c_c", "b", "w",
      "pi_base", "pi_pert"
    ),
    include = TRUE,
    data = list(
      N = length(simulated_data[[i]]$t),
      t = simulated_data[[i]]$t,
      RR = simulated_data[[i]]$RR,
      N_sin = 25,
      freqs = freq_bands
    ),
    iter = 10000, warmup = 5000,
    chains = 5, cores = 5,
    seed = 12345,
    control = list(adapt_delta = 0.99, ## Target acceptance rate
                   max_treedepth = 20) ## Maximum per-side steps (before U-turn)
  )

  # Save fitted object to disk for reproducibility
  saveRDS(model_fit, file = paste0("models/model_fit_scenario_",i,".RDS"))
}

