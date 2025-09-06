# ---
# R-R Interval Time Series Simulation Generator
#
# Description:
# This script generates synthetic R-R interval (RRi) data based on the generative
# model structure described in the accompanying Stan model. It allows for the
# specification of ground-truth parameters to simulate different physiological
# scenarios, providing a "gold standard" for model validation and comparison
# with traditional analysis methods (e.g., windowed analysis, STFT).
#
# The generation process follows these key steps:
#   1. Define underlying dynamic trajectories for mean RRi and SDNN using
#      double-logistic functions.
#   2. Define dynamic spectral proportions (VLF, LF, HF power distribution)
#      controlled by a master logistic controller.
#   3. Synthesize the structured oscillatory signal (S_t) from a sum of
#      sinusoids with a 1/f^b power law.
#   4. Invert the variance equation to calculate the time-varying amplitude (A_t)
#      that ensures the structured signal variance matches the target.
#   5. Combine the baseline, the structured signal, and a residual white noise
#      component to produce the final RRi time series.
#
# Author: Matías Castillo-Aguilar
# ---

# 1. --- Load necessary libraries ---
# Using pacman to install/load libraries for cleaner setup
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tibble, ggplot2, tidyr, cowplot)

# 2. --- Helper Functions ---

#' Calculate a standard logistic growth curve (sigmoid).
#'
#' @param t A numeric vector of time points.
#' @param location The midpoint (center) of the sigmoid.
#' @param rate The growth rate or steepness of the curve.
#' @return A numeric vector of values between 0 and 1.
logistic_curve <- function(t, location, rate) {
  1 / (1 + exp(-rate * (t - location)))
}

#' Generate synthetic R-R interval data for one scenario.
#'
#' This function encapsulates the entire generative process of the model.
#' @param params A list containing all necessary ground-truth parameters.
#' @param t_vec A numeric vector of time points for the simulation (in minutes).
#' @param freqs A list of numeric vectors for VLF, LF, and HF band frequencies (in Hz).
#' @param N_sin The number of sinusoids per frequency band.
#' @return A tibble containing the full simulation output, including the final
#'   RRi series and all intermediate ground-truth trajectories.
generate_rri_simulation <- function(params, t_vec, freqs, N_sin) {

  N <- length(t_vec)
  t_sec <- t_vec * 60 # Convert time to seconds for Hz frequencies

  # --- 1. Define the shared logistic components (D1, D2) ---
  D1 <- logistic_curve(t_vec, params$tau, params$lambda)
  D2 <- logistic_curve(t_vec, params$tau + params$delta, params$phi)

  # --- 2. Build the ground-truth baseline and SDNN trajectories ---
  RR_baseline_true <- params$alpha_r - params$beta_r * D1 + params$c_r * params$beta_r * D2
  SDNN_t_true <- params$alpha_s - params$beta_s * D1 + params$c_s * params$beta_s * D2

  # --- 3. Build the master controller C(t) and spectral proportions p(t) ---
  C_t <- D1 * (1 - params$c_c * D2)
  # Mix baseline proportions (pi_base) and perturbation proportions (pi_pert)
  # The result is an N x 3 matrix of proportions over time
  p_t_true <- (1 - C_t) %*% t(params$pi_base) + C_t %*% t(params$pi_pert)
  colnames(p_t_true) <- c("p_vlf", "p_lf", "p_hf")

  # --- 4. Synthesize the spectral oscillators S_j(t) ---
  S_t_matrix <- matrix(0, nrow = N, ncol = 3)
  colnames(S_t_matrix) <- c("S_vlf", "S_lf", "S_hf")

  for (j in 1:3) {
    # Amplitude law (1/f^b noise structure)
    a_k <- exp(-params$b / 2 * log(freqs[[j]]))

    # Generate random coefficients for sine and cosine components
    # This replaces the non-centered parameterization from the Stan model for generation
    u_sin <- rnorm(N_sin, 0, sd = params$sigma_u[j] * a_k)
    u_cos <- rnorm(N_sin, 0, sd = params$sigma_u[j] * a_k)

    # Precompute sin/cos templates (computationally intensive part)
    T_mat <- outer(t_sec, freqs[[j]], "*")
    sin_mat <- sin(2 * pi * T_mat)
    cos_mat <- cos(2 * pi * T_mat)

    # Synthesize the j-th oscillator and de-mean it
    S_j_unnorm <- sin_mat %*% u_sin + cos_mat %*% u_cos
    S_t_matrix[, j] <- S_j_unnorm - mean(S_j_unnorm)
  }

  # --- 5. Derive the internal amplitude A(t) using inversion ---
  # This is the key step that links SDNN(t) to the spectral components
  var_struct_true <- SDNN_t_true^2 * params$w
  var_resid_true <- SDNN_t_true^2 * (1 - params$w)

  # Covariance of the basis signals S
  Sigma_S <- cov(S_t_matrix)

  # Calculate the denominator for the A(t) inversion
  # This is equivalent to sqrt(rows_dot_product(p_t * Sigma_S, p_t))
  denom_sq <- rowSums((p_t_true %*% Sigma_S) * p_t_true)
  denom <- sqrt(denom_sq)

  # Calculate the time-varying amplitude A(t)
  A_t_true <- sqrt(var_struct_true) / denom
  # Handle potential division by zero if a band has no power
  A_t_true[!is.finite(A_t_true)] <- 0

  # --- 6. Combine for the final signal mu and add residual noise ---
  # Calculate the structured part of the signal
  sum_weighted_S <- rowSums(S_t_matrix * p_t_true)
  mu_true <- RR_baseline_true + A_t_true * sum_weighted_S

  # Add the residual (unstructured) noise to get the final RRi series
  RR_observed <- rnorm(N, mean = mu_true, sd = sqrt(var_resid_true))

  # --- 7. Assemble and return results ---
  tibble(
    time = t_vec,
    RR_observed = RR_observed,
    RR_baseline_true = RR_baseline_true,
    SDNN_t_true = SDNN_t_true,
    A_t_true = A_t_true,
    mu_true = mu_true,
    var_struct_true = var_struct_true,
    var_resid_true = var_resid_true
  ) %>%
    bind_cols(as_tibble(p_t_true))
}


# 3. --- Simulation Setup ---

# Define shared simulation parameters
SIM_DURATION_MIN <- 15 # Total duration in minutes
SAMPLING_RATE_HZ <- 4  # Sampling rate for RRi series (4 Hz is typical)
N_points <- SIM_DURATION_MIN * 60 * SAMPLING_RATE_HZ
time_vector <- seq(0, SIM_DURATION_MIN, length.out = N_points)

# Define frequency bands (VLF, LF, HF)
N_SINUSOIDS <- 50 # Number of sinusoids to approximate spectrum in each band
freq_bands <- list(
  vlf = seq(0.003, 0.039, length.out = N_SINUSOIDS),
  lf  = seq(0.040, 0.149, length.out = N_SINUSOIDS),
  hf  = seq(0.150, 0.400, length.out = N_SINUSOIDS)
)

# --- Define Parameters for the Three Scenarios ---

# Scenario 1: Classic Sympatho-Vagal Response
# A sharp drop in RR/SDNN with a partial recovery, accompanied by a shift
# from high-frequency (HF) to low-frequency (LF) power and back.
params1 <- list(
  # Double-logistic timing
  tau = 6, delta = 3, lambda = 3, phi = 2,
  # RR(t) params
  alpha_r = 950, beta_r = 250, c_r = 1.0,
  # SDNN(t) params
  alpha_s = 60, beta_s = 40, c_s = 1.0,
  # p(t) params
  pi_base = c(0.1, 0.2, 0.7), # VLF, LF, HF - Rest (HF dominant)
  pi_pert = c(0.5, 0.3, 0.2), # VLF, LF, HF - Stress (LF dominant)
  c_c = 0.8,
  # Spectral & Noise params
  b = 1.0, sigma_u = c(1, 1, 1) * 0.1, w = 0.90 # 90% structured variance
)

# Scenario 2: Incomplete Recovery with Spectral Persistence
# A physiological response where recovery is not complete, leaving a lasting
# spectral signature of the perturbation.
params2 <- list(
  # Double-logistic timing
  tau = 6, delta = 3, lambda = 3, phi = 2,
  # RR(t) params - c_r < 1 means incomplete mean recovery
  alpha_r = 950, beta_r = 250, c_r = 0.5,
  # SDNN(t) params - c_s < 1 means incomplete variability recovery
  alpha_s = 60, beta_s = 40, c_s = 0.6,
  # p(t) params - c_c < 1 means spectral signature persists
  pi_base = c(0.1, 0.2, 0.7),
  pi_pert = c(0.1, 0.7, 0.2),
  c_c = 0.4,
  # Spectral & Noise params
  b = 1.0, sigma_u = c(1, 1, 1) * 0.1, w = 0.90
)

# Scenario 3: High Noise with a Stable Spectrum
# A scenario with high overall variability but where the spectral balance
# remains constant. This tests the model's ability to separate structured
# variance from random noise.
params3 <- list(
  # Double-logistic timing (less dramatic transition)
  tau = 8, delta = 4, lambda = 1.5, phi = 1.5,
  # RR(t) params
  alpha_r = 900, beta_r = 300, c_r = 1.0,
  # SDNN(t) params - High baseline variability
  alpha_s = 100, beta_s = 20, c_s = 1.0,
  # p(t) params - Stable spectrum (base and pert are similar)
  pi_base = c(0.1, 0.45, 0.45),
  pi_pert = c(0.1, 0.6, 0.3),
  c_c = 1.0,
  # Spectral & Noise params - w is lower, so more residual noise
  b = 1.0, sigma_u = c(1, 1, 1) * 0.1, w = 0.60 # 60% structured variance
)


# 4. --- Generate and Visualize Data ---
# This block demonstrates how to use the function and visualize the output.
if (interactive()) {

  # Set a seed for reproducibility
  set.seed(123)

  # --- Generate data for Scenario 1 ---
  sim_data <- generate_rri_simulation(params1, time_vector, freq_bands, N_SINUSOIDS)

  # --- Create Plots to Visualize the Ground Truth and Simulated Data ---

  # Plot 1: Observed RRi and underlying true mean (mu)
  p1 <- ggplot(sim_data, aes(x = time)) +
    geom_line(aes(y = RR_observed), color = "grey70", alpha = 0.8) +
    geom_line(aes(y = mu_true), color = "firebrick") +
    labs(title = "A) Observed RRi and True Mean Signal (μ)",
         x = "Time (minutes)", y = "RR Interval (ms)") +
    scale_x_continuous(expand = c(0,0)) +
    theme_cowplot(font_size = 12)

  # Plot 2: Ground-truth time-domain dynamics
  p2 <- pivot_longer(sim_data, c("RR_baseline_true", "SDNN_t_true")) |>
    ggplot(aes(x = time, y = value, color = name)) +
    facet_wrap(~ name, ncol = 2, scales = "free_y") +
    geom_line(linewidth = 1) +
    scale_color_manual(values = c("dodgerblue", "darkorange")) +
    labs(title = "B) Ground-Truth Time-Domain Trajectories",
         x = "Time (minutes)", y = "Value (ms)", color = "Metric") +
    scale_x_continuous(expand = c(0,0)) +
    theme_cowplot(font_size = 12) +
    theme(legend.position = "bottom")

  # Plot 3: Ground-truth spectral proportion dynamics
  p3 <- sim_data %>%
    select(time, starts_with("p_")) %>%
    pivot_longer(cols = -time, names_to = "Band", values_to = "proportion") %>%
    mutate(Band = factor(toupper(gsub("p_", "", Band)), levels = c("VLF", "LF", "HF"))) %>%
    ggplot(aes(x = time, y = proportion, fill = Band)) +
    geom_area(alpha = 0.8) +
    scale_fill_manual(values = c("VLF" = "#882255", "LF" = "#44AA99", "HF" = "#DDCC77")) +
    labs(title = "C) Ground-Truth Spectral Proportions p(t)",
         x = "Time (minutes)", y = "Proportion of Power", fill = "Band") +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    theme_cowplot(font_size = 12) +
    theme(legend.position = "bottom")

  # Combine plots into a single figure
  plot_grid(p1, p2, p3, ncol = 1)
}

# To generate data for other scenarios, simply change the params list:
# sim_data_scen2 <- generate_rri_simulation(params2, time_vector, freq_bands, N_SINUSOIDS)
# sim_data_scen3 <- generate_rri_simulation(params3, time_vector, freq_bands, N_SINUSOIDS)
