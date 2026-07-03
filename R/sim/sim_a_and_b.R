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
SIM_DURATION_MIN <- 12 # Total duration in minutes
SAMPLING_RATE_HZ <- 2  # Sampling rate for RRi series (4 Hz is typical)
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
params <- function(good_hrv = TRUE) {
  if(good_hrv) {
    lambda = 3; phi = 3; tau = 6; delta = 1.5;
    beta_r = -400; c_r = 0.9; c_s = 1.0; c_c = 1;
  } else {
    lambda = 2; phi = 1; tau = 6; delta = 1.5;
    beta_r = -400; c_r = 0.7; c_s = 0.6; c_c = 0.6;
  }

  list(
    # Double-logistic timing
    lambda = lambda, phi = phi, tau = tau, delta = delta,
    # RR(t) params
    alpha_r = rnorm(1, 750, 50), beta_r = beta_r, c_r = c_r,
    # SDNN(t) params
    alpha_s = 50, beta_s = -45, c_s = c_s,
    # Spectral & Noise params
    c_c = c_c, w = 0.80, # 90% structured variance
    # p(t) params
    pi_base = c(0.40, 0.30, 0.30), # VLF, LF, HF - Rest (HF dominant)
    pi_pert = c(0.60, 0.30, 0.10), # VLF, LF, HF - Stress (LF dominant)
    rho_gp = c(0.5,0.5,0.5)
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
    seed = 1
  )$data

sim_data_bad <-
  generate_rri_simulation(
    N = N_points,
    t_max = SIM_DURATION_MIN,
    params = params(FALSE),
    N_sin = N_SINUSOIDS,
    seed = 2
  )$data

sim_data <- list(A = sim_data_good,
                 B = sim_data_bad) |>
  rbindlist(idcol = "Subject")

# Plot 1: Observed RRi and underlying true mean (mu)
ggplot(sim_data, aes(t, RR, col = Subject)) +
  facet_grid(rows = vars(Subject)) +
  geom_line(aes(y = RR), color = "gray", linewidth = 1/2, show.legend = FALSE) +
  geom_line(aes(y = RR_baseline), linewidth = 2, show.legend = FALSE) +
  scale_color_manual(values = c(A = "dodgerblue", B = "firebrick")) +
  labs(x = "Tiempo (minutos)", y = NULL) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0.1,0), breaks = NULL) +
  theme_classic(base_size = 15) +
  theme(strip.background = element_blank(),
        strip.text = element_blank())
