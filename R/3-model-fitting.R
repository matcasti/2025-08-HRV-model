
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
      "c_c_logit", "w_logit","alpha_gp","rho_gp",
      "y_base_log", "y_pert_log",
      "lambda","phi","tau","delta",
      "alpha_r","beta_r","c_r",
      "alpha_s","beta_s","c_s",
      "c_c", "w",
      "pi_base", "pi_pert"
    ),
    include = TRUE,
    data = list(
      N = length(simulated_data[[i]]$data$t),
      t = simulated_data[[i]]$data$t,
      RR = simulated_data[[i]]$data$RR,
      N_sin = 20,
      freqs = list(
        seq(0.003, 0.039, length.out = 20), # VLF
        seq(0.040, 0.149, length.out = 20), # LF
        seq(0.150, 0.400, length.out = 20)  # HD
      ),
      lambda_mu = params[[i]]$lambda,
      phi_mu = params[[i]]$phi,
      tau_mu = params[[i]]$tau,
      delta_mu = params[[i]]$delta
    ),
    iter = 10000, warmup = 5000,
    chains = 4, cores = 4,
    seed = 12345,
    control = list(adapt_delta = 0.95, ## Target acceptance rate
                   max_treedepth = 10) ## Maximum per-side steps (before U-turn)
  )

  # Save fitted object to disk for reproducibility
  saveRDS(model_fit, file = paste0("models/model_fit_scenario_",i,".RDS"))
}
