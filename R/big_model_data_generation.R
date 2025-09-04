# --------------------------------------------------------------------------
# Generative Model for R-R Interval Simulation
# --------------------------------------------------------------------------

# === 1. Model Parameters ===
# All parameters are centralized here for easy tweaking.

t_start <- 0 # Start time in minutes
t_end <- 15 # End time in minutes

# --- Overall Dynamics ---
## D_{1,2}(t)

D_1 <- function(t) {
  1 / (1 + exp(-lambda * (t - tau)))
}

D_2 <- function(t) {
  1 / (1 + exp(-phi * (t - tau - delta)))
}

lambda <- 3  # Rate of exercise-induced perturbations
phi <- 2     # Rate of post-exercise recovery
tau <- 6     # Onset of perturbations
delta <- 3   # Offset to recovery

## RR(t)
alpha_r <- 800   # Mean R-R interval in ms
beta_r <- 400    # Exercise-induced drop in ms
c_r <- 0.8       # Proportion recovery

## SDNN(t)
alpha_s <- 50    # Baseline R-R variability in ms
beta_s <- 25     # Exercise-induced drop in ms
c_s <- 1.2       # Proportion recovery

## C(t)
c_c <- 0.8      # Proportion recovery

## Fractional SDNN
w <- 0.8

# --- Proportional Band Dynamics (p_j(t)) ---
# Baseline state (e.g., rest, high HF)
pi_base <- c(VLF = 0.1, LF = 0.3, HF = 0.6)
# Perturbed state (e.g., stress, high LF)
pi_pert <- c(VLF = 0.7, LF = 0.2, HF = 0.1)

# --- Intra-Band Spectral Content (S_j(t)) ---
# Frequency band definitions
band_defs <- list(
  VLF = list(min = 0.003, max = 0.039),
  LF  = list(min = 0.040,  max = 0.149),
  HF  = list(min = 0.150,  max = 0.4)
)
N_sin <- 25 # Number of sine waves per band
b <- 1 # Spectral exponent (1.0 for pink noise)


# === 2. Simulation Core ===
# --- Time Vector ---
t <- seq(t_start, t_end, length.out = 1200)
n_points <- length(t)

# --- Generate Time-Varying components ---
## Mean R-R interval: RR(t)
RR_t <- alpha_r - beta_r * D_1(t) + c_r * beta_r * D_2(t)
plot(t, RR_t, type = "l"); grid()

## Standard deviation of RRi: SDNN(t)
SDNN_t <- alpha_s - beta_s * D_1(t) + c_s * beta_s * D_2(t)
plot(t, SDNN_t, type = "l"); grid()

## Master spectral function: C(t)
C_t <- D_1(t) * (1 - c_c * D_2(t))
plot(t, C_t, type = "l"); grid()

# Calculate the time-varying proportion matrix p_j
p_j <- (1 - C_t) %*% t(pi_base) + C_t %*% t(pi_pert)
matplot(t, p_j, lty = 1, type = "l"); grid()
colnames(p_j) <- c("VLF", "LF", "HF")

# Total Amplitude: A(t)
 var_structured <- SDNN_t^2 * w
 var_noise <- SDNN_t^2 * (1 - w)

A_t <- sqrt(var_structured) / sqrt(rowSums(p_j^2))
plot(t, A_t, type = "l"); grid()

# --- Generate Spectral Band Signals S_j(t) ---
S_t_matrix <- matrix(0, nrow = n_points, ncol = 3)
colnames(S_t_matrix) <- c("VLF", "LF", "HF")

set.seed(123) # for reproducibility

for (j in 1:3) {
  band_name <- names(band_defs)[j]
  f_min <- band_defs[[j]]$min
  f_max <- band_defs[[j]]$max

  # 1. Define frequencies and random phases
  f_k <- seq(f_min, f_max, length.out = N_sin)
  phi_k <- runif(N_sin, 0, 2 * pi)

  # 2. Calculate amplitudes based on power law
  a_k <- f_k^(-b / 2)

  # 3. Create signal by summing sine waves
  # outer(t, f_k) creates a matrix where element (i, j) is t_i * f_k
  # This is an efficient way to calculate all sine values at once.
  sine_waves <- sin(2 * pi * outer(t * 60, f_k) + matrix(phi_k, nrow = n_points, ncol = N_sin, byrow = TRUE))

  # 4. Weight sines by their amplitudes and sum them up
  S_j_unnormalized <- sine_waves %*% a_k

  # 5. Normalize to have unit variance and zero mean
  S_t_matrix[, j] <- scale(S_j_unnormalized)
}

# --- Combine All Components for the Final RR Signal ---
# Element-wise multiplication of band signals by their proportions
weighted_S <- S_t_matrix * p_j

# Sum the weighted band signals
sum_weighted_S <- rowSums(weighted_S)

# Final R-R signal generation
mu <- RR_t + A_t * sum_weighted_S
RRi_t <- rnorm(length(t), mu, sqrt(var_noise))

plot(t, RRi_t, type = "l")

# -------------------------------------------------------------------------

rr_min <- min(RRi_t);
rr_range <- max(RRi_t) - rr_min;
t_min <- min(t);
t_range <- max(t) - t_min;

# -------------------------------------------------------------------------


# === 3. Visualization ===
graphics.off() # Close any open plot windows
layout(matrix(1:4, ncol = 2), heights = c(0.5, 0.5), widths = c(0.5, 0.5))
par(mar = c(4, 4, 2, 1))

# Plot 1: Final R-R Interval Time Series
plot(t, RRi_t, type = 'l', col = "navy",
     main = "Simulated R-R Interval Time Series",
     xlab = "Time (s)", ylab = "R-R Interval (ms)")
grid()

# Plot 2: Total Amplitude A(t)
matplot(t, weighted_S, type = 'l',
     main = "Weighted Sine Sum",
     xlab = "Time (s)", ylab = "ms")
grid()
legend(9, y = 1.5, legend = colnames(p_j), col = 1:3, lty = 1:3, lwd = 2)

# Plot 3: SDNN SDNN(t)
plot(t, SDNN_t, type = 'l', col = "darkred", lwd = 2,
     main = "Time-Varying SDNN SDNN(t)",
     xlab = "Time (s)", ylab = "SDNN (ms)")
grid()

# Plot 4: Proportional Contributions p_j(t)
matplot(t, p_j, type = 'l', lty = 1, lwd = 2,
        main = "Time-Varying Proportions p(t)",
        xlab = "Time (s)", ylab = "Proportion", ylim = c(0,0.8))
grid()
legend(9, y = 0.8, legend = colnames(p_j), col = 1:3, lty = 1, lwd = 2)

par(mfrow = c(1,1))

sim_data <- data.frame(t, RRi_t)

plot(sim_data, type = "l")

library(rstan)

model <- rstan::stan_model(file = "models/rri_model.stan")

rr_t_fit <- rstan::sampling(
  object = model,
  pars = c(
    "lambda_log","phi_log","tau_logit","delta_logit",
    "alpha_r_logit","beta_r_logit","c_r_logit",
    "alpha_s_logit","beta_s_logit","c_s_logit",
    "c_c_logit", "b_log", "w_logit",
    "y_base_log", "y_pert_log",
    "lambda","phi","tau","delta",
    "alpha_r","beta_r","c_r",
    "alpha_s","beta_s","c_s",
    "c_c", "b", "w",
    "pi_base", "pi_pert"
  ),
  include = TRUE,
  data = list(N = length(sim_data$t),
              t = sim_data$t,
              RR = sim_data$RRi_t,
              N_sin = N_sin,
              freqs = list(
                seq(0.003, 0.039, length.out = N_sin),
                seq(0.040, 0.149, length.out = N_sin),
                seq(0.150, 0.400, length.out = N_sin)
              ),
              phases = list(
                runif(N_sin, 0, 2 * pi),
                runif(N_sin, 0, 2 * pi),
                runif(N_sin, 0, 2 * pi)
              )),
  iter = 10000, warmup = 8000,
  chains = 5, cores = 5,
  seed = 12345,
  control = list(adapt_delta = 0.99,
                 max_treedepth = 10)
)

saveRDS(rr_t_fit, file = "models/rr_t_fit.rds")

rstan::check_hmc_diagnostics(rr_t_fit)

## Tiempo que se demora el muestreo
rstan::get_elapsed_time(rr_t_fit) |>
  rowSums() |>
  max() |>
  (\(x) cat("=== Time elapsed ===\n",
            " minutes: ", x/60,
            "\n hours: ",x/3600,
            "\n====================",
            sep = ""))()

posterior_rr <- extract(rr_t_fit) |>
  data.table::as.data.table()

local({
  png(file = "rplot.png",width=25,height=25,units="in",res=600);
  plot(posterior_rr,pch=".");
  dev.off()
})

posterior_epred <- posterior_rr[, lapply(.SD, median)]

# shinystan::launch_shinystan(rr_t_fit)
