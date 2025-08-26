# --------------------------------------------------------------------------
# Generative Model for R-R Interval Simulation
# --------------------------------------------------------------------------

# === 1. Model Parameters ===
# All parameters are centralized here for easy tweaking.

t_start <- 0 # Start time in seconds
t_end <- 15 # End time in seconds

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
beta_s <- 20     # Exercise-induced drop in ms
c_s <- 1.2       # Proportion recovery

## C(t)
c_c <- 0.8      # Proportion recovery

# --- Proportional Band Dynamics (p_j(t)) ---
# Baseline state (e.g., rest, high HF)
pi_base <- c(VLF = 0.1, LF = 0.3, HF = 0.6)
# Perturbed state (e.g., stress, high LF)
pi_pert <- c(VLF = 0.7, LF = 0.2, HF = 0.1)

# --- Intra-Band Spectral Content (S_j(t)) ---
# Frequency band definitions
band_defs <- list(
  VLF = list(min = 0.003, max = 0.04),
  LF  = list(min = 0.04,  max = 0.15),
  HF  = list(min = 0.15,  max = 0.4)
)
N_sin <- 30 # Number of sine waves per band
beta <- 1 # Spectral exponent (1.0 for pink noise)


# === 2. Simulation Core ===
# --- Time Vector ---
t <- seq(t_start, t_end, length.out = 2000)
n_points <- length(t)

# --- Generate Time-Varying components ---
## Mean R-R interval: RR(t)
RR_t <- alpha_r - beta_r * D_1(t) + c_r * beta_r * D_2(t)
plot(t, RR_t, type = "l"); grid()

## Standard deviation of RRi: SDNN(t)
SDNN_t <- alpha_s - beta_s * D_1(t) + c_s * beta_s * D_2(t)
plot(t, SDNN_t, type = "l"); grid()

## Master spectral function: C(t)
C_t <- D_1(t) - c_c * D_2(t)
plot(t, C_t, type = "l"); grid()

## Proportion functions: p_j(t)
p_j <- (1 - C_t) %*% t(pi_base) + C_t %*% t(pi_pert)
matplot(p_j, lty = 1, type = "l"); grid()

# Total Amplitude: A(t)
A_t <- SDNN_t / sqrt(rowSums(p_j^2))
plot(t, A_t, type = "l"); grid()

# Calculate the time-varying proportion matrix p_t
p_t <- (1 - C_t) %*% t(pi_base) +
  C_t %*% t(pi_pert)
colnames(p_t) <- c("VLF", "LF", "HF")


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
  a_k <- f_k^(-beta / 2)

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
RRi_t <- RR_t + A_t * sum_weighted_S


# === 3. Visualization ===
graphics.off() # Close any open plot windows
layout(matrix(1:4, ncol = 2), heights = c(0.5, 0.5), widths = c(0.5, 0.5))
par(mar = c(4, 4, 2, 1), bg = "white")

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
legend(10, y = 1.5, legend = colnames(p_t), col = 1:3, lty = 1:3, lwd = 2)

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
legend(10, y = 0.8, legend = colnames(p_t), col = 1:3, lty = 1, lwd = 2)

sim_data <- data.frame(t, RRi_t)

plot(sim_data, type = "l")

library(rstan)


model <- rstan::stan_model(file = "models/rri_model.stan")


rr_t_fit <- sampling(
  object = model,
  pars = c(
    "lambda","phi","tau","delta",
    "alpha_r","beta_r","c_r",
    "alpha_s","beta_s","c_s",
    "c_c", "pi_pert", "pi_base",
    "b", "sigma"
  ),
  include = TRUE,
  data = list(N = length(sim_data$t),
              t = sim_data$t,
              RR = sim_data$RRi_t,
              N_sin = 50,
              freqs = list(
                seq(0.003, 0.04, length.out = 50),
                seq(0.04, 0.15, length.out = 50),
                seq(0.15, 0.4, length.out = 50)
              ),
              phases = list(
                runif(50, 0, 2 * pi),
                runif(50, 0, 2 * pi),
                runif(50, 0, 2 * pi)
              )),
  iter = 10000, warmup = 5000,
  chains = 4, cores = 4,
  seed = 1234
)

saveRDS(rr_t_fit, file = "models/rr_t_fit.rds")
