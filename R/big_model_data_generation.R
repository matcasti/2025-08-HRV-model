# --------------------------------------------------------------------------
# Generative Model for R-R Interval Simulation (R version)
# Mirrors the Stan inference process
# --------------------------------------------------------------------------

# === 1. Model Parameters ===
# --------------------------------------------------------------------------
# All fixed parameters centralized here for clarity

t_start <- 0      # Start time (minutes)
t_end   <- 15     # End time (minutes)

# --- Logistic dynamics ---
# Exercise onset (D1) and recovery (D2)
lambda <- 3   # Rate of exercise-induced perturbations
phi    <- 2   # Rate of post-exercise recovery
tau    <- 6   # Onset of perturbations (minutes)
delta  <- 3   # Offset to recovery (minutes)

D_1 <- function(t) { 1 / (1 + exp(-lambda * (t - tau))) }
D_2 <- function(t) { 1 / (1 + exp(-phi * (t - tau - delta))) }

# --- R-R mean trajectory ---
alpha_r <- 800   # Baseline RR (ms)
beta_r  <- 400   # Drop during exercise (ms)
c_r     <- 0.8   # Recovery proportion

# --- R-R variability (SDNN) trajectory ---
alpha_s <- 50    # Baseline SDNN (ms)
beta_s  <- 25    # Drop during exercise (ms)
c_s     <- 1.2   # Recovery proportion

# --- Spectral transition control ---
c_c <- 0.8       # Recovery proportion for spectral mixing

# --- Variance decomposition ---
w <- 0.9         # Fraction of structured variance
# (w close to 1 → mostly structured oscillatory variance)

# --- Band mixing proportions ---
pi_base <- c(VLF = 0.1, LF = 0.3, HF = 0.6)  # Resting
pi_pert <- c(VLF = 0.7, LF = 0.2, HF = 0.1)  # Perturbed (exercise)

# --- Spectral bands ---
band_defs <- list(
  VLF = list(min = 0.003, max = 0.039),
  LF  = list(min = 0.040, max = 0.149),
  HF  = list(min = 0.150, max = 0.400)
)
N_sin <- 25   # Sine waves per band
b     <- 1    # Spectral exponent (1 → pink noise)

# === 2. Time Vector ===
# --------------------------------------------------------------------------
t <- seq(t_start, t_end, length.out = 2400)  # Simulation grid
n_points <- length(t)


# === 3. Deterministic Trajectories ===
# --------------------------------------------------------------------------

# Mean R-R interval (logistic drop + partial recovery)
RR_t <- alpha_r - beta_r * D_1(t) + c_r * beta_r * D_2(t)

# Standard deviation of RR intervals (SDNN_t)
SDNN_t <- alpha_s - beta_s * D_1(t) + c_s * beta_s * D_2(t)

# Spectral mixing curve C(t)
C_t <- D_1(t) * (1 - c_c * D_2(t))

# Time-varying band proportions p_j(t)
p_j <- (1 - C_t) %*% t(pi_base) + C_t %*% t(pi_pert)
colnames(p_j) <- c("VLF", "LF", "HF")


# === 4. Spectral Band Signals S_j(t) ===
# --------------------------------------------------------------------------
# Construct stochastic signals for each band using sine-wave mixtures

S_t_matrix <- matrix(0, nrow = n_points, ncol = 3)
colnames(S_t_matrix) <- c("VLF", "LF", "HF")

set.seed(123)  # For reproducibility of random phases

phases_stan <- vector("list", 3)
freqs_stan <- vector("list", 3)
for (j in 1:3) {
  f_min <- band_defs[[j]]$min
  f_max <- band_defs[[j]]$max

  # Frequencies (linearly spaced in each band)
  freqs_stan[[j]] <- f_k <- seq(f_min, f_max, length.out = N_sin)

  # Random phases
  phases_stan[[j]] <- phi_k <- runif(N_sin, 0, 2 * pi)

  # Amplitudes with power-law scaling
  a_k <- f_k^(-b / 2)

  # Generate sine waves for this band
  sine_waves <- sin(
    2 * pi * outer(t * 60, f_k) +
      matrix(phi_k, nrow = n_points, ncol = N_sin, byrow = TRUE)
  )

  # Weighted sum of sine waves, centered at 0
  S_t_matrix[, j] <- sine_waves %*% a_k - mean(sine_waves %*% a_k)
}

# Covariance of band signals (used for scaling)
Sigma_S <- cov(S_t_matrix)


# === 5. Variance Partition and Scaling ===
# --------------------------------------------------------------------------

# Structured vs unstructured variance components
var_structured <- SDNN_t^2 * w
var_noise      <- SDNN_t^2 * (1 - w)

# Denominator: quadratic form p' Σ p
denom_sq <- rowSums((p_j %*% Sigma_S) * p_j)

# Amplitude scaling factor A(t)
A_t <- sqrt(var_structured) / sqrt(denom_sq)


# === 6. Final RR Signal Generation ===
# --------------------------------------------------------------------------

# Weighted spectral contributions
weighted_S <- S_t_matrix * p_j
sum_weighted_S <- rowSums(weighted_S)

# Deterministic + structured component
mu <- RR_t + A_t * sum_weighted_S

# Add Gaussian residuals for unstructured noise
RRi_t <- rnorm(n_points, mean = mu, sd = sqrt(var_noise))

# --------------------------------------------------------------------------
# 7. Diagnostics: Does simulated RRi_t recover the target SDNN_t?
# --------------------------------------------------------------------------

# Function: rolling standard deviation
roll_sd <- function(x, w) {
  n <- length(x)
  sds <- rep(NA, n)
  half <- floor(w / 2)
  for (i in (half + 1):(n - half)) {
    sds[i] <- sd(x[(i - half):(i + half)])
  }
  return(sds)
}

# Empirical SDNN from simulated RRi_t
SDNN_emp <- roll_sd(RRi_t - RR_t, 100)

# Compare theoretical vs empirical
par(mfrow = c(1,1))
plot(t, SDNN_t, type = "l", col = "blue", lwd = 2,
     main = "Target vs Empirical SDNN(t)",
     ylab = "SDNN (ms)", xlab = "Time (min)", ylim = c(10, 70))
lines(t, SDNN_emp, col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Theoretical SDNN(t)", "Empirical (rolling)"),
       col = c("blue", "red"), lty = c(1,2), lwd = 2, bty = "n")
grid()

# --------------------------------------------------------------------------
# 9. Visualization of Simulation Components
# --------------------------------------------------------------------------

graphics.off() # Close any open plot windows

# Define a 2x2 layout for four diagnostic plots
layout(matrix(1:4, ncol = 2), heights = c(0.5, 0.5), widths = c(0.5, 0.5))
par(mar = c(4, 4, 2, 1)) # margins for plots (bottom, left, top, right)

# --- Plot 1: Final simulated RR interval time series ---
plot(t, RRi_t, type = 'l', col = "navy",
     main = "Simulated R-R Interval Time Series",
     xlab = "Time (s)", ylab = "R-R Interval (ms)")
grid()

# --- Plot 2: Weighted band contributions (spectral sum before amplitude scaling) ---
matplot(t, weighted_S, type = 'l',
        main = "Weighted Sine Sum",
        xlab = "Time (s)", ylab = "ms")
grid()
legend(9, y = 1.5, legend = colnames(p_j), col = 1:3, lty = 1:3, lwd = 2, cex = 0.7)

# --- Plot 3: Target SDNN(t) trajectory (deterministic) ---
plot(t, SDNN_t, type = 'l', col = "darkred", lwd = 2,
     main = "Time-Varying SDNN(t)",
     xlab = "Time (s)", ylab = "SDNN (ms)")
grid()

# --- Plot 4: Time-varying proportional contributions p_j(t) ---
matplot(t, p_j, type = 'l', lty = 1, lwd = 2,
        main = "Time-Varying Proportions p(t)",
        xlab = "Time (s)", ylab = "Proportion", ylim = c(0,0.8))
grid()
legend(9, y = 0.8, legend = colnames(p_j), col = 1:3, lty = 1, lwd = 2, cex = 0.7)

par(mfrow = c(1,1)) # Reset plotting layout

# --------------------------------------------------------------------------
# 10. Bayesian Inference via Stan
# --------------------------------------------------------------------------

# Compile the Stan model
library(rstan)
model <- rstan::stan_model(file = "models/rri_model.stan")

## Obtain point estimates
point_est <- rstan::optimizing(object = model,
                  iter = 50000,
                  data = list(
                    N = length(t),
                    t = t,
                    RR = RRi_t,
                    N_sin = N_sin,
                    freqs = freqs_stan
                  ), hessian = TRUE)$par

point_est[grep("^D|sin|cos|^RR_|^SDNN_|^C_t|^p_t|^S_|^var_|^A_|^M|^denom|^sum|mu|Sigma", names(point_est), value = TRUE, invert = TRUE)]

# Run HMC sampling
# Note:
# - High iterations (10,000) with long warmup (8,000) for stability
# - 5 chains in parallel (cores = 5)
# - adapt_delta raised to 0.99 to reduce divergences
# - max_treedepth = 10 for controlling exploration
rr_t_fit <- rstan::sampling(
  object = model,
  pars = c(
    "lambda_log","phi_log","tau_logit","delta_logit",
    "alpha_r_logit","beta_r_logit","c_r_logit",
    "alpha_s_logit","beta_s_logit","c_s_logit",
    "c_c_logit", "b_log", "w_logit", "sigma_u",
    "y_base_log", "y_pert_log",
    "lambda","phi","tau","delta",
    "alpha_r","beta_r","c_r",
    "alpha_s","beta_s","c_s",
    "c_c", "b", "w",
    "pi_base", "pi_pert"
  ),
  include = TRUE,
  data = list(
    N = length(t),
    t = t,
    RR = RRi_t,
    N_sin = N_sin,
    freqs = freqs_stan
  ),
  iter = 10000, warmup = 5000,
  chains = 5, cores = 5,
  seed = 12345,
  control = list(adapt_delta = 0.80,
                 max_treedepth = 10)
)

# Save fitted object to disk for reproducibility
saveRDS(rr_t_fit, file = "models/rr_t_fit.rds")

# --------------------------------------------------------------------------
# 11. Diagnostics and Posterior Inspection
# --------------------------------------------------------------------------

# Run built-in HMC diagnostics (divergences, treedepth, etc.)
rstan::check_hmc_diagnostics(rr_t_fit)

# Print maximum elapsed time across chains
rstan::get_elapsed_time(rr_t_fit) |>
  rowSums() |>
  max() |>
  (\(x) cat("=== Time elapsed ===\n",
            " minutes: ", x/60,
            "\n hours: ",x/3600,
            "\n====================",
            sep = ""))()

# Extract posterior samples into a data.table for further analysis
posterior_rr <- extract(rr_t_fit) |>
  data.table::as.data.table()

# Save pairs plot of posterior parameters (warning: large and dense plot)
local({
  png(file = "rplot.png", width = 25, height = 25, units = "in", res = 600)
  plot(posterior_rr, pch = ".")
  dev.off()
})


posterior_epred <- posterior_rr[, lapply(.SD, median)]

# shinystan::launch_shinystan(rr_t_fit)
