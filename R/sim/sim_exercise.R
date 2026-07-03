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
SIM_DURATION_MIN <- 15 # Total duration in minutes
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
params <- function() {
  list(
    # Double-logistic timing
    lambda = 2, phi = 4, tau = 6, delta = 3,
    # RR(t) params
    alpha_r = 800, beta_r = -300, c_r = 0.8,
    # SDNN(t) params
    alpha_s = 50, beta_s = -50, c_s = 0.8,
    # Spectral & Noise params
    c_c = 0.8, w = 0.90, # 90% structured variance
    # p(t) params
    pi_base = c(0.40, 0.30, 0.30), # VLF, LF, HF - Rest (HF dominant)
    pi_pert = c(0.60, 0.20, 0.10), # VLF, LF, HF - Stress (LF dominant)
    rho_gp = c(1,1,1)
  )
}

# -------------------------------------------------------------------------

# 3. --- Generate and Visualize Data ---
# This block demonstrates how to use the function and visualize the output.

# --- Generate data for Scenario ---
sim_data <-
  generate_rri_simulation(
    N = N_points,
    t_max = SIM_DURATION_MIN,
    params = params(),
    N_sin = N_SINUSOIDS,
    seed = 123
  )$data

legend <- NA
# --- Create Plots to Visualize the Ground Truth and Simulated Data ---

# Plot 1: Observed RRi and underlying true mean (mu)
p1 <- ggplot(sim_data, aes(x = t)) +
  geom_line(aes(y = RR), color = "purple", linewidth = 1/2, alpha = 0.5) +
  geom_point(aes(y = RR), color = "purple", cex = 0.5, alpha = 1) +
  labs(x = NULL, y = NULL) +
  scale_x_continuous(expand = c(0,0.1), breaks = NULL) +
  theme_void(base_size = 15)

p1

ggplot(sim_data, aes(x = t)) +
  geom_line(aes(y = RR), color = "gray", linewidth = 1/2, alpha = 0.5) +
  geom_point(aes(y = RR), color = "gray", cex = 0.5, alpha = 0.5) +
  geom_line(aes(y = RR_baseline), color = "purple", linewidth = 2) +
  labs(x = NULL, y = NULL) +
  scale_x_continuous(expand = c(0,0.1), breaks = NULL) +
  theme_void(base_size = 15)


