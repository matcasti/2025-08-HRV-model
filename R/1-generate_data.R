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

# 1. --- Load necessary libraries and scripts ---
library(data.table)
library(ggplot2)
source("R/_functions.R")

# 2. --- Simulation Setup ---

# Define shared simulation parameters
SIM_DURATION_MIN <- 15 # Total duration in minutes
SAMPLING_RATE_HZ <- 2  # Sampling rate for RRi series (4 Hz is typical)
N_points <- SIM_DURATION_MIN * 60 * SAMPLING_RATE_HZ
time_vector <- seq(0, SIM_DURATION_MIN, length.out = N_points)

# Define frequency bands (VLF, LF, HF)
N_SINUSOIDS <- 25 # Number of sinusoids to approximate spectrum in each band
freq_bands <- list(
  vlf = seq(0.003, 0.039, length.out = N_SINUSOIDS),
  lf  = seq(0.040, 0.149, length.out = N_SINUSOIDS),
  hf  = seq(0.150, 0.400, length.out = N_SINUSOIDS)
)

params <- vector("list", length = 3)

# --- Define Parameters for the Three Scenarios ---
# Scenario 1: Classic Sympatho-Vagal Response
# A sharp drop in RR/SDNN with a partial recovery, accompanied by a shift
# from high-frequency (HF) to low-frequency (LF) power and back.
params[[1]] <- list(
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
  b = 1.0, w = 0.90 # 90% structured variance
)

# Scenario 2: Incomplete Recovery with Spectral Persistence
# A physiological response where recovery is not complete, leaving a lasting
# spectral signature of the perturbation.
params[[2]] <- list(
  # Double-logistic timing
  tau = 6, delta = 3, lambda = 3, phi = 2,
  # RR(t) params - c_r < 1 means incomplete mean recovery
  alpha_r = 950, beta_r = 250, c_r = 0.5,
  # SDNN(t) params - c_s < 1 means incomplete variability recovery
  alpha_s = 60, beta_s = 40, c_s = 0.6,
  # p(t) params - c_c < 1 means spectral signature persists
  pi_base = c(0.1, 0.2, 0.7), # VLF, LF, HF - Rest
  pi_pert = c(0.7, 0.2, 0.1), # VLF, LF, HF - Stress
  c_c = 0.4,
  # Spectral & Noise params
  b = 1.0, w = 0.90
)

# Scenario 3: High Noise with a Stable Spectrum
# A scenario with high overall variability but where the spectral balance
# remains constant. This tests the model's ability to separate structured
# variance from random noise.
params[[3]] <- list(
  # Double-logistic timing (less dramatic transition)
  tau = 6, delta = 3, lambda = 1.5, phi = 1.5,
  # RR(t) params
  alpha_r = 900, beta_r = 300, c_r = 1.0,
  # SDNN(t) params - High baseline variability
  alpha_s = 100, beta_s = 20, c_s = 1.0,
  # p(t) params - Stable spectrum (base and pert are similar)
  pi_base = c(0.1, 0.45, 0.45), # VLF, LF, HF - Rest
  pi_pert = c(0.2, 0.5, 0.3), # VLF, LF, HF - Stress
  c_c = 1.0,
  # Spectral & Noise params - w is lower, so more residual noise
  b = 1.0, w = 0.60 # 60% structured variance
)

## Pull everything together in a nice table for the paper
params_table <- lapply(`names<-`(params, paste("Scenario", 1:3)), function(i) {
  char_params <- lapply(i, function(j) {
    if (length(j) > 1) {
      paste0("[",paste0(j, collapse = ", "),"]")
    } else {
      format(j, digits = 1, nsmall = 2)
    }
  })
  as.data.table(char_params)
}) |> rbindlist(idcol = "Scenario") |>
  transpose(keep.names = "Parameter", make.names = "Scenario") |>
  knitr::kable()

simulated_data <- vector("list", length = 3)

# 3. --- Generate and Visualize Data ---
# This block demonstrates how to use the function and visualize the output.
if (interactive()) {

  plots <- vector("list", length = 3)

  for(i in 1:3) {
    # --- Generate data for Scenario i ---
    simulated_data[[i]] <-
      sim_data <-
      generate_rri_simulation(
        N = N_points,
        t_max = SIM_DURATION_MIN,
        params = params[[i]],
        N_sin = N_SINUSOIDS,
        seed = 123
      )

    legend <- FALSE
    if (i == 3) {
      legend <- NA
    }
    # --- Create Plots to Visualize the Ground Truth and Simulated Data ---

    # Plot 1: Observed RRi and underlying true mean (mu)
    p1 <- ggplot(sim_data, aes(x = t)) +
      geom_line(aes(y = RR, color = "Observed"), alpha = 0.8, show.legend = legend) +
      geom_line(aes(y = mu, color = "True µ(t)"), show.legend = legend) +
      scale_color_manual(values = c("Observed" = "grey70", "True µ(t)" = "firebrick")) +
      labs(subtitle = ifelse(i==1,"Observed RRi Signal",""),
           x = "Time (minutes)", y = "ms",
           color = "Signal") +
      scale_x_continuous(expand = c(0,0), name = NULL, breaks = NULL) +
      theme_classic(base_size = 12)

    # Plot 2: Ground-truth time-domain dynamics
    p2 <- sim_data[, c("t","RR_baseline", "SDNN_t")] |>
      ggplot(aes(x = t)) +
      geom_ribbon(aes(ymin = RR_baseline - SDNN_t,
                      ymax = RR_baseline + SDNN_t,
                      fill = "SDNN"), show.legend = legend) +
      geom_line(aes(y = RR_baseline, color = "Mean R-R"), linewidth = 1, show.legend = legend) +
      scale_color_manual(values = c("Mean R-R" = "darkred")) +
      scale_fill_manual(values = c("SDNN" = "pink")) +
      labs(subtitle = ifelse(i==1,"Time-domain dynamics",""),
           x = "Time (minutes)", y = "ms",
           color = "Line", fill = "Shaded area") +
      scale_x_continuous(expand = c(0,0), name = NULL, breaks = NULL) +
      theme_classic(base_size = 12) +
      theme(legend.position = "right")

    # Plot 3: Ground-truth spectral proportion dynamics
    p3 <- melt(sim_data,
               id = "t",
               measure.vars = c("p_vlf","p_lf","p_hf")
    )[, variable := factor(variable,
                           levels = c("p_vlf","p_lf","p_hf"),
                           labels = c("VLF","LF","HF"))][] |>
      ggplot(aes(x = t, y = value, fill = variable, color = variable)) +
      geom_area(alpha = 0.8, show.legend = legend) +
      scale_fill_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                        aesthetics = c("fill", "color")) +
      labs(subtitle = ifelse(i==1,"Spectral signatures",""),
           x = "Time (minutes)", y = "Proportion of Power", fill = "Band", color = "Band") +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0)) +
      theme_classic(base_size = 12) +
      theme(legend.position = "right")

    # Combine plots into a single figure
    plots[[i]] <- ggpubr::ggarrange(p1, p2, p3, ncol = 1, align = "v")
  }

  fig <- ggpubr::ggarrange(plotlist = plots,
                           ncol = 3,
                           widths = c(2,2,3),
                           align = "hv",
                           labels = c("(A)","(B)","(C)"))

  ggsave(filename = "figures/fig-generated-data.svg", fig,
         device = "svg", width = 9, height = 9)
  ggsave(filename = "figures/fig-generated-data.pdf", fig,
         device = "pdf", width = 9, height = 9)

  saveRDS(simulated_data, file = "data/simulated_data.RDS")
  saveRDS(params, file = "data/simulation_parameters.RDS")
  saveRDS(freq_bands, file = "data/freq_bands.RDS")
}
