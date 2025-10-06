# Title: Generate Synthetic RRi Data with Asymmetric (Exponential) Recovery
# Description: This script creates a simulated RR-interval time series
#              that follows a logistic drop followed by a slower,
#              exponential recovery. This is intended for testing model
#              robustness to misspecification of the recovery function.

library(data.table)
library(rstan)
library(ggplot2)

source("R/_functions.R")

# --- 1. Define Helper Functions ---

# Standard logistic (sigmoid) function
logistic_curve <- function(t, location, rate) {
  1 / (1 + exp(-rate * (t - location)))
}

# Asymmetric exponential recovery curve (shaped like a CDF)
exponential_recovery <- function(t, start_time, rate) {
  # Time since the recovery event started
  time_since_start <- pmax(0, t - start_time)
  # Cumulative exponential function, scaled from 0 to 1
  1 - exp(-rate * time_since_start)
}


# --- 2. Set Simulation Parameters ---

# Time parameters
N      <- 1200   # Number of data points (e.g., 20 minutes at 1 Hz)
t      <- seq(from = 0, to = 15, length.out = N) # Time vector in minutes
tau    <- 5      # Time of perturbation event (minutes)
delta  <- 1      # Delay between perturbation and recovery onset (minutes)
lambda <- 5      # Rate/steepness of the logistic perturbation
phi    <- 0.8    # Rate of the exponential recovery (lower = slower recovery)

# RRi amplitude parameters
alpha  <- 1000   # Initial baseline RRi (ms)
beta   <- -300    # Magnitude of the RRi drop (ms)
c_r    <- 0.90   # Fractional recovery (0.9 = 90% recovery of the drop)

# Noise parameter
sigma  <- 15     # Standard deviation of the residual noise (ms)


# --- 3. Generate the Time Series ---

# Create the perturbation and recovery building blocks
perturbation <- logistic_curve(t, location = tau, rate = lambda)
recovery     <- exponential_recovery(t, start_time = tau + delta, rate = phi)

# Combine the components to create the true mean RRi trajectory
# The logic follows the manuscript's model: Baseline - Drop + Recovery
mean_rri <- alpha + (beta * perturbation) - (c_r * beta * recovery)

# Add Gaussian noise to create the final observed signal
set.seed(123) # for reproducibility
observed_rri <- rnorm(n = N, mean = mean_rri, sd = sigma)

# --- 4. Package and Save Data ---

# Store the results in a data frame
sim_data <- data.frame(
  time = t,
  true_mean_rri = mean_rri,
  observed_rri = observed_rri
)

# Optional: Save the data to a CSV file for use with your Stan model
# write.csv(sim_data, "asymmetric_recovery_data.csv", row.names = FALSE)

# --- 5. Visualize the Result ---

# Plot the generated data to confirm it looks correct
plot(sim_data$time, sim_data$observed_rri,
     type = "o", pch = 16, cex = 0.5,
     xlab = "Time (minutes)", ylab = "RR Interval (ms)",
     main = "Simulated Data with Asymmetric (Exponential) Recovery",
     ylim = range(sim_data$observed_rri) * c(0.95, 1.05))

lines(sim_data$time, sim_data$true_mean_rri,
      col = "firebrick", lwd = 2.5)

legend("topright",
       legend = c("Observed RRi (with noise)", "True Mean RRi Trajectory"),
       col = c("black", "firebrick"),
       pch = c(16, NA),
       lty = c(NA, 1),
       lwd = c(NA, 2.5),
       bty = "n")


# -------------------------------------------------------------------------

if (!file.exists("models/model_fit_asymetric.RDS")) {

  # Prepare data ------------------------------------------------------------
  N_sin <- 30
  stan_data <- list(
    N = length(sim_data$time),
    t = sim_data$time,
    RR = sim_data$observed_rri,
    N_sin = N_sin,
    freqs = list(
      seq(0.003, 0.039, length.out = N_sin), # VLF
      seq(0.040, 0.149, length.out = N_sin), # LF
      seq(0.150, 0.400, length.out = N_sin)  # HF
    ),
    lambda_mu = lambda,
    phi_mu = phi,
    tau_mu = tau,
    delta_mu = delta
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
    seed = 123,
    control = list(adapt_delta = 0.95, ## Target acceptance rate
                   max_treedepth = 10) ## Maximum per-side steps (before U-turn)
  )
  saveRDS(model_fit, file = "models/model_fit_asymetric.RDS")
} else {
  model_fit <- readRDS(file = "models/model_fit_asymetric.RDS")
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

fig_obs <- ggplot(sim_data, aes(time, true_mean_rri)) +
  geom_line(linetype = 1, col = "dodgerblue", linewidth = 2) +
  labs(subtitle = "True underlying RRi trend",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(limits = c(650, 1100)) +
  theme_classic(base_size = 12)

fig_mu <- ggplot() +
  geom_line(aes(time, observed_rri, col = "Observed"), sim_data, linetype = 1) +
  geom_ribbon(aes(x = sim_data$time, fill = "Estimated µ(t)",
                  ymin = mu - hdi,
                  ymax = mu + hdi),
              data = mu_hat, alpha = 0.5) +
  geom_line(aes(x = sim_data$time, y = mu, col = "Estimated µ(t)"),
            data = mu_hat, linewidth = 1) +
  scale_color_manual(values = c("Observed" = "gray",
                                "Estimated µ(t)" = "firebrick"),
                     aesthetics = c("fill", "color")) +
  labs(fill = "Line", col = "Line", subtitle = "Observed and reconstructed signal",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(limits = c(650, 1100)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

fig <- cowplot::plot_grid(fig_obs, fig_mu, align = "hv")

ggsave(filename = "figures/fig-asymetric.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-asymetric.pdf", fig,
       device = "pdf", width = 9, height = 9)
