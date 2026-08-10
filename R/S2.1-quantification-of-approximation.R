
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(CardioCurveR)
library(rstan)
library(ggplot2)

## Import functions
source("R/_functions.R")

## Load model
model_fit <- readRDS(file = "models/model_fit_prior_normal.RDS")

## Load the data
poc_data <- import_RRi_txt(file = "data-raw/rri-jabf.txt",
                           remove_ectopic = TRUE,
                           filter_noise = FALSE) |>
  as.data.table()

# Prepare data ------------------------------------------------------------

## Get an initial estimation of logistic parameters
prior_params <- with(poc_data, estimate_RRi_curve(time, RRi))$parameters |> abs()

N_sin <- 50

## Stan data
stan_data <- list(
  N = length(poc_data$time),
  t = poc_data$time,
  RR = poc_data$RRi,
  N_sin = N_sin,
  freqs = list(
    seq(0.003, 0.039, length.out = N_sin), # VLF
    seq(0.040, 0.149, length.out = N_sin), # LF
    seq(0.150, 0.400, length.out = N_sin)  # HF
  ),
  lambda_mu = prior_params[["lambda"]],
  phi_mu = prior_params[["phi"]],
  tau_mu = prior_params[["tau"]],
  delta_mu = prior_params[["delta"]]
)

# -------------------------------------------------------------------------

# 1. Recreate the Transformed Data structures
t <- stan_data$t
N <- length(t)
N_sin <- stan_data$N_sin
freqs <- stan_data$freqs

log_freqs_scaled <- list()
sin_mat <- list()
cos_mat <- list()
diag_sum <- list()

norm_fact <- 1.0 / (N - 1)

for(j in 1:3) {
  # Scale log-frequencies for the GP
  lf <- log(freqs[[j]])
  log_freqs_scaled[[j]] <- (lf - min(lf)) / (max(lf) - min(lf))

  # Create centered basis functions
  T_mat <- outer(t * 60, freqs[[j]])
  sin_raw <- sin(2 * pi * T_mat)
  cos_raw <- cos(2 * pi * T_mat)

  s_mat <- sweep(sin_raw, 2, colMeans(sin_raw), "-")
  c_mat <- sweep(cos_raw, 2, colMeans(cos_raw), "-")

  sin_mat[[j]] <- s_mat
  cos_mat[[j]] <- c_mat

  # Calculate expected diagonal contribution
  g_sin_diag <- colSums(s_mat^2) * norm_fact
  g_cos_diag <- colSums(c_mat^2) * norm_fact
  diag_sum[[j]] <- g_sin_diag + g_cos_diag
}

# 2. Extract posterior draws
draws <- rstan::extract(model_fit)
n_draws <- nrow(draws$rho_gp)

ratios <- numeric(n_draws)
mean_diags <- numeric(n_draws)
max_off_diags <- numeric(n_draws)

# 3. Reconstruct signals and compute metrics for each draw
for(i in 1:n_draws) {
  S_t <- matrix(0, nrow = N, ncol = 3)

  for(j in 1:3) {
    # Reconstruct GP covariance matrix
    x <- log_freqs_scaled[[j]]
    D <- outer(x, x, "-")^2
    rho <- draws$rho_gp[i, j]
    K <- exp(-0.5 * D / rho^2)
    diag(K) <- diag(K) + 1e-8

    # Non-centered parameterization transformation
    L <- t(chol(K)) # t() converts R's upper Cholesky to Stan's lower Cholesky
    z_gp_vec <- as.numeric(draws$z_gp[i, j, ])
    log_v <- as.numeric(L %*% z_gp_vec)
    a_k <- exp(log_v)

    # Scale coefficients
    base_v_diag <- sum(a_k^2 * diag_sum[[j]])
    full_scale <- a_k / sqrt(base_v_diag + 1e-12)

    u_sin <- draws$z_sin[i, j, ] * full_scale
    u_cos <- draws$z_cos[i, j, ] * full_scale

    # Synthesize the mean-centered oscillator
    S_j <- as.numeric(sin_mat[[j]] %*% u_sin + cos_mat[[j]] %*% u_cos)
    S_t[, j] <- S_j - mean(S_j)
  }

  # Compute covariance metrics
  cov_mat <- cov(S_t)
  diag_vals <- diag(cov_mat)
  off_diag_vals <- cov_mat[upper.tri(cov_mat)]

  mean_diags[i] <- mean(abs(diag_vals))
  max_off_diags[i] <- max(abs(off_diag_vals))
  ratios[i] <- max_off_diags[i] / mean_diags[i]
}

# 4. Summarize results
ratio_summary <- data.table(
  Mean   = mean(ratios),
  SD     = sd(ratios),
  Median = median(ratios),
  Minimum = min(ratios),
  `Lower bound` = quantile(ratios, 0.025),
  `Upper bound` = quantile(ratios, 0.975),
  Maximum = max(ratios)
) |> transpose(keep.names = "stat")

ratio_summary <- ratio_summary[, list(Metric = stat, Value = round(V1, 3))]

knitr::kable(ratio_summary, align = "l")

#> |Metric      |Value |
#> |:-----------|:-----|
#> |Mean        |0.003 |
#> |SD          |0.001 |
#> |Median      |0.002 |
#> |Minimum     |0.000 |
#> |Lower bound |0.001 |
#> |Upper bound |0.006 |
#> |Maximum     |0.017 |
