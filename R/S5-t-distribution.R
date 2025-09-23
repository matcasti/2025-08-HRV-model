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
library(rstan)
source("R/_functions.R")

# 2. --- Simulation Setup ---

# Define shared simulation parameters
SIM_DURATION_MIN <- 15 # Total duration in minutes
SAMPLING_RATE_HZ <- 2  # Sampling rate for RRi series
N_points <- SIM_DURATION_MIN * 60 * SAMPLING_RATE_HZ
time_vector <- seq(0, SIM_DURATION_MIN, length.out = N_points)

# Define frequency bands (VLF, LF, HF)
N_SINUSOIDS <- 20 # Number of sinusoids to approximate spectrum in each band
freq_bands <- list(
  vlf = seq(0.003, 0.039, length.out = N_SINUSOIDS),
  lf  = seq(0.040, 0.149, length.out = N_SINUSOIDS),
  hf  = seq(0.150, 0.400, length.out = N_SINUSOIDS)
)

# --- Define Parameters for the Three Scenarios ---
# Scenario 1: Classic Sympatho-Vagal Response
# A sharp drop in RR/SDNN with a partial recovery, accompanied by a shift
# from high-frequency (HF) to low-frequency (LF) power and back.
params <- list(
  # Double-logistic timing
  lambda = 2, phi = 3, tau = 6, delta = 3,
  # RR(t) params
  alpha_r = 800, beta_r = 400, c_r = 1.0,
  # SDNN(t) params
  alpha_s = 50, beta_s = 40, c_s = 1.0,
  # Spectral & Noise params
  c_c = 0.8, w = 0.80, # 90% structured variance
  # p(t) params
  pi_base = c(0.2, 0.2, 0.6), # VLF, LF, HF - Rest (HF dominant)
  pi_pert = c(0.4, 0.4, 0.2), # VLF, LF, HF - Stress (LF dominant)
  rho_gp = c(1, 1, 1) * 0.1
)

# -------------------------------------------------------------------------


# 3. --- Generate and Visualize Data ---
# This block demonstrates how to use the function and visualize the output.

# --- Generate data for Scenario i ---
sim_data <-
  generate_rri_simulation(
    N = N_points,
    t_max = SIM_DURATION_MIN,
    params = params,
    N_sin = N_SINUSOIDS,
    seed = 12345,
    t_dist = TRUE,
    nu = 3
  )

legend <- NA
# --- Create Plots to Visualize the Ground Truth and Simulated Data ---

# Plot 1: Observed RRi and underlying true mean (mu)
p1 <- ggplot(sim_data$data, aes(x = t)) +
  geom_line(aes(y = RR, color = "Observed"), linewidth = 1/3, show.legend = legend) +
  geom_line(aes(y = mu, color = "True µ(t)"), linewidth = 1/3, show.legend = legend) +
  scale_color_manual(values = c("Observed" = "grey70", "True µ(t)" = "firebrick")) +
  labs(subtitle = "Observed RRi Signal",
       x = "Time (minutes)", y = "ms",
       color = "Signal") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12)

# Plot 2: Ground-truth time-domain dynamics
p2 <- sim_data$data[, c("t","RR_baseline", "SDNN_t")] |>
  ggplot(aes(x = t)) +
  geom_ribbon(aes(ymin = RR_baseline - SDNN_t,
                  ymax = RR_baseline + SDNN_t,
                  fill = "SDNN"), show.legend = legend) +
  geom_line(aes(y = RR_baseline, color = "Mean R-R"), linewidth = 1, show.legend = legend) +
  scale_color_manual(values = c("Mean R-R" = "darkred")) +
  scale_fill_manual(values = c("SDNN" = "pink")) +
  labs(subtitle = "Time-domain dynamics",
       x = "Time (minutes)", y = "ms",
       color = "Line", fill = "Shaded area") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

# Plot 3: Ground-truth spectral proportion dynamics
p3 <- melt(sim_data$data,
           id = "t",
           measure.vars = c("p_vlf","p_lf","p_hf")
)[, variable := factor(variable,
                       levels = c("p_vlf","p_lf","p_hf"),
                       labels = c("VLF","LF","HF"))][] |>
  ggplot(aes(x = t, y = value, fill = variable, color = variable)) +
  geom_area(alpha = 0.8, show.legend = legend) +
  scale_fill_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                    aesthetics = c("fill", "color")) +
  labs(subtitle = "Spectral signatures",
       x = "Time (minutes)", y = "Proportion of Power", fill = "Band", color = "Band") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

# Combine plots into a single figure
fig_left <- ggpubr::ggarrange(p1, p2, p3, ncol = 1, align = "v")


# -------------------------------------------------------------------------

if (!file.exists("models/model_t_dist.RDS")) {

  # Prepare data ------------------------------------------------------------
  N_sin <- 20
  stan_data <- list(
    N = length(sim_data$data$t),
    t = sim_data$data$t,
    RR = sim_data$data$RR,
    N_sin = N_SINUSOIDS,
    freqs = list(
      seq(0.003, 0.039, length.out = N_SINUSOIDS), # VLF
      seq(0.040, 0.149, length.out = N_SINUSOIDS), # LF
      seq(0.150, 0.400, length.out = N_SINUSOIDS)  # HF
    ),
    lambda_mu = params$lambda,
    phi_mu = params$phi,
    tau_mu = params$tau,
    delta_mu = params$delta
  )

  # Fit the model -----------------------------------------------------------

  model <- stan_model(file = "models/rri_model.stan")
  model_fit <- rstan::sampling(
    object = model,
    init = 0,
    pars = c(
      "lambda","phi","tau","delta",
      "alpha_r","beta_r","c_r",
      "alpha_s","beta_s","c_s",
      "c_c", "w", "pi_base", "pi_pert",
      "mu","RR_baseline","SDNN_t"
    ),
    data = stan_data,
    iter = 10000, warmup = 5000,
    chains = 4, cores = 4,
    seed = 12345,
    control = list(adapt_delta = 0.95, ## Target acceptance rate
                   max_treedepth = 10) ## Maximum per-side steps (before U-turn)
  )
  saveRDS(model_fit, file = "models/model_t_dist.RDS")
} else {
  model_fit <- readRDS(file = "models/model_t_dist.RDS")
}

# Reconstructed signal ----------------------------------------------------

mu_hat <- extract(model_fit, pars = "mu") |>
  as.data.table()

mu_hat <- transpose(mu_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

mu_hat[, time := gsub("mu.V", "", time) |> as.numeric()]
mu_hat[, draw := gsub("V", "", draw) |> as.numeric()]

mu_hat <- mu_hat[, list(mu = median(value), hdi = diff(ggdist::hdi(value)[1,])), keyby = time]

# Generate plot -----------------------------------------------------------

fig_up <- ggplot() +
  geom_line(aes(t, RR, col = "Observed"), sim_data$data, linetype = 1) +
  geom_ribbon(aes(x = sim_data$data$t, fill = "Estimated µ(t)",
                  ymin = mu - hdi,
                  ymax = mu + hdi),
              data = mu_hat, alpha = 0.5) +
  geom_line(aes(x = sim_data$data$t, y = mu, col = "Estimated µ(t)"),
            data = mu_hat, linewidth = 1/2) +
  scale_color_manual(values = c("Observed" = "gray",
                                "Estimated µ(t)" = "firebrick"),
                     aesthetics = c("fill", "color")) +
  labs(fill = "Line", col = "Line", subtitle = "Observed and reconstructed signal",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous() +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

sim <- as.data.table(extract(model_fit))[, lambda:pi_pert.V3]
sim[, row_id := seq_len(.N)]
sim_long <- melt(sim, id.vars = "row_id")

levels(sim_long$variable) <- c("lambda","phi","tau","delta",
                               "alpha[r]", "beta[r]", "c[r]",
                               "alpha[s]", "beta[s]", "c[s]", "c[c]", "w",
                               "pi[base]~VLF", "pi[base]~LF", "pi[base]~HF",
                               "pi[pert]~VLF", "pi[pert]~LF", "pi[pert]~HF")

params <- unlist(params)[1:18] |>
  as.data.table(keep.rownames = TRUE) |>
  `names<-`(c("variable", "value"))

params$variable <- levels(sim_long$variable)
params$variable <- factor(params$variable, levels = params$variable)


fig_low <- ggplot(sim_long, aes(x = value, fill = variable)) +
  facet_wrap(~variable, ncol = 3, scales = "free_x", labeller = label_parsed) +
  ggdist::stat_halfeye(normalize = "panels", show.legend = FALSE,
                       adjust = 2) +
  geom_vline(data = params, aes(xintercept = value)) +
  scale_y_continuous(breaks = NULL, labels = NULL) +
  labs(x = "Parameter value", y = "Density") +
  scale_fill_viridis_d(option = "C", begin = 0.1, alpha = 0.9) +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank())

fig <- cowplot::plot_grid(fig_up, fig_low, align = "hv", nrow = 2,
                          rel_heights = c(0.1,0.15), axis = "l")

ggsave(filename = "figures/fig-student-t.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-student-t.pdf", fig,
       device = "pdf", width = 9, height = 9)
