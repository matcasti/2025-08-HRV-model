# ---
# R-R Interval Time Series Simulation Generator
#
# Description:
# This script generates synthetic R-R interval (RRi) data based on the generative
# model structure described in the accompanying Stan model. It allows for the
# specification of ground-truth parameters to simulate different physiological
# scenarios, providing a "gold standard" for model validation and comparison
# with traditional analysis methods (e.g., windowed analysis, STFT).
# ---

# 1. --- Load necessary libraries and scripts ---
library(data.table)
library(ggplot2)
library(gt)
source("R/_functions.R")

# 2. --- Simulation Setup ---

# Define shared simulation parameters
SIM_DURATION_MIN <- 5 # Total duration in minutes
SAMPLING_RATE_HZ <- 2  # Sampling rate for RRi series (4 Hz is typical)
N_points <- SIM_DURATION_MIN * 60 * SAMPLING_RATE_HZ
time_vector <- seq(0, SIM_DURATION_MIN, length.out = N_points)

# Define frequency bands (VLF, LF, HF)
N_SINUSOIDS <- 30 # Number of sinusoids to approximate spectrum in each band
freq_bands <- list(
  vlf = seq(0.003, 0.039, length.out = N_SINUSOIDS),
  lf  = seq(0.040, 0.149, length.out = N_SINUSOIDS),
  hf  = seq(0.150, 0.400, length.out = N_SINUSOIDS)
)

# --- Define Parameters for the Three Scenarios ---
# Scenario 1: Classic Sympatho-Vagal Response
# A sharp drop in RR/SDNN with a partial recovery, accompanied by a shift
# from high-frequency (HF) to low-frequency (LF) power and back.
params <- function(good_hrv = TRUE) {
  if(good_hrv) {
    alpha_s <- 60
    pi_base <- c(0.40, 0.30, 0.30)
  } else {
    alpha_s <- 30
    pi_base <- c(0.60, 0.20, 0.10)
  }
  pi_pert <- pi_base

  list(
    # Double-logistic timing
    lambda = 2, phi = 3, tau = 6, delta = 3,
    # RR(t) params
    alpha_r = 800, beta_r = 0, c_r = 1.0,
    # SDNN(t) params
    alpha_s = alpha_s, beta_s = 0, c_s = 1.0,
    # Spectral & Noise params
    c_c = 0, w = 0.90, # 90% structured variance
    # p(t) params
    pi_base = pi_base, # VLF, LF, HF - Rest (HF dominant)
    pi_pert = pi_pert, # VLF, LF, HF - Stress (LF dominant)
    rho_gp = c(1,1,1)
  )
}

# -------------------------------------------------------------------------

# 3. --- Generate and Visualize Data ---
# This block demonstrates how to use the function and visualize the output.

# --- Generate data for Scenario ---
sim_data_good <-
  generate_rri_simulation(
    N = N_points,
    t_max = SIM_DURATION_MIN,
    params = params(TRUE),
    N_sin = N_SINUSOIDS,
    seed = 123
  )$data

sim_data_bad <-
  generate_rri_simulation(
    N = N_points,
    t_max = SIM_DURATION_MIN,
    params = params(FALSE),
    N_sin = N_SINUSOIDS,
    seed = 123
  )$data

legend <- NA
# --- Create Plots to Visualize the Ground Truth and Simulated Data ---

# Plot 1: Observed RRi and underlying true mean (mu)
p1 <- ggplot(sim_data_good, aes(x = t)) +
  geom_line(aes(y = RR), color = "dodgerblue", linewidth = 1/2, show.legend = legend) +
  geom_hline(yintercept = 800, linetype = 2, linewidth = 1/2, color = "gray20") +
  labs(x = NULL, y = NULL) +
  scale_x_continuous(expand = c(0,0.1), breaks = NULL) +
  scale_y_continuous(limits = c(600, 1000)) +
  theme_classic(base_size = 15)

p2 <- ggplot(sim_data_bad, aes(x = t)) +
  geom_line(aes(y = RR), color = "firebrick", linewidth = 1/2, show.legend = legend) +
  geom_hline(yintercept = 800, linetype = 2, linewidth = 1/2, color = "gray20") +
  labs(x = "Tiempo (minutos)", y = NULL) +
  scale_x_continuous(expand = c(0,0.1)) +
  scale_y_continuous(limits = c(600, 1000)) +
  theme_classic(base_size = 15)

fig <- cowplot::plot_grid(p1, p2, nrow = 2, rel_heights = c(0.9,1))

