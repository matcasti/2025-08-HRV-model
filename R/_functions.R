#' Generate a synthetic R-R interval time series from constrained parameters
#'
#' This function encapsulates the full data generating process from the Stan model.
#' All model parameters are provided on their constrained, physiologically meaningful
#' scale.
#'
#' @param N Number of data points.
#' @param t_max Maximum time in minutes.
#' @param N_sin Number of sinusoids per frequency band.
#' @param params Named list with model parameters.
#' @param seed An integer for reproducibility.
#'
#' @return A data frame containing the time vector 't', the final generated
#'   'RR' series, the underlying mean 'mu', and other key components.
generate_rri_simulation <- function(N,
                                    t_max,
                                    params, # Note: Expects GP params now (alpha_gp, rho_gp)
                                    N_sin,
                                    seed = 123) {

  # Helper function to compute the GP's squared exponential covariance kernel
  gp_exp_quad_cov <- function(x, alpha, rho) {
    N <- length(x)
    K <- matrix(0, N, N)
    for (i in 1:N) {
      for (j in 1:N) {
        K[i, j] <- alpha^2 * exp(-0.5 * ((x[i] - x[j]) / rho)^2)
      }
    }
    return(K)
  }

  # Set seed for complete reproducibility
  set.seed(seed)

  # --- 1. Define Core Dynamic Functions ---
  # These are the double-logistic building blocks for the model's dynamics.
  D_1 <- function(t) { 1 / (1 + exp(-params$lambda * (t - params$tau))) }
  D_2 <- function(t) { 1 / (1 + exp(-params$phi * (t - params$tau - params$delta))) }

  # Define the simulation time grid
  t <- seq(0, t_max, length.out = N)

  # --- 2. Generate Time-Varying Trajectories ---
  # These define the evolution of the signal's mean, total variability, and spectral mix.
  RR_t <- params$alpha_r - params$beta_r * D_1(t) + params$c_r * params$beta_r * D_2(t)
  SDNN_t <- params$alpha_s - params$beta_s * D_1(t) + params$c_s * params$beta_s * D_2(t)
  C_t <- D_1(t) * (1 - params$c_c * D_2(t))

  # Time-varying band proportions p_j(t) via convex combination
  p_j <- (1 - C_t) %*% t(params$pi_base) + C_t %*% t(params$pi_pert)
  colnames(p_j) <- c("VLF", "LF", "HF")

  # --- 3. Pre-computation of Basis Functions (like Stan's transformed data) ---
  band_defs <- list(
    VLF = c(0.003, 0.039),
    LF  = c(0.040, 0.149),
    HF  = c(0.150, 0.400)
  )

  freqs_list <- lapply(band_defs, function(b) seq(b[1], b[2], length.out = N_sin))
  log_freqs_list <- lapply(freqs_list, log)

  # Precompute sine/cosine basis matrices and Gram matrices
  sin_mat_list <- vector("list", 3)
  cos_mat_list <- vector("list", 3)
  G_sin_list <- vector("list", 3)
  G_cos_list <- vector("list", 3)
  G_sin_cos_list <- vector("list", 3)

  for (j in 1:3) {
    T_mat <- outer(t * 60, freqs_list[[j]])
    sin_mat_list[[j]] <- sin(2 * pi * T_mat)
    cos_mat_list[[j]] <- cos(2 * pi * T_mat)

    normalization <- 1.0 / (N - 1)
    G_sin_list[[j]] <- t(sin_mat_list[[j]]) %*% sin_mat_list[[j]] * normalization
    G_cos_list[[j]] <- t(cos_mat_list[[j]]) %*% cos_mat_list[[j]] * normalization
    G_sin_cos_list[[j]] <- t(sin_mat_list[[j]]) %*% cos_mat_list[[j]] * normalization
  }

  # --- 4. Generate Spectral Components via GP (like Stan's transformed parameters) ---
  # This is the core of the new generative process.

  # A. Simulate the standard normal deviates (the "z" parameters)
  z_gp <- replicate(3, rnorm(N_sin), simplify = FALSE)
  z_sin <- replicate(3, rnorm(N_sin), simplify = FALSE)
  z_cos <- replicate(3, rnorm(N_sin), simplify = FALSE)

  # B. Loop through bands to generate coefficients
  u_sin_list <- vector("list", 3)
  u_cos_list <- vector("list", 3)
  log_v_list <- vector("list", 3)

  for (j in 1:3) {
    # Step 1: Generate the smooth spectral envelope from the GP
    K <- gp_exp_quad_cov(log_freqs_list[[j]], params$alpha_gp[j], params$rho_gp[j])
    K <- K + diag(1e-8 * params$alpha_gp[j]^2 + 1e-12, N_sin) # Scaled jitter
    L <- t(chol(K)) # Lower triangular Cholesky factor
    log_v <- L %*% z_gp[[j]]

    log_v_list[[j]] <- log_v

    # Step 2: Normalize the GP output to have unit expected variance
    a_k <- exp(log_v)
    diag_sum <- diag(G_sin_list[[j]]) + diag(G_cos_list[[j]])
    base_v_diag <- sum((a_k^2) * diag_sum)
    full_scale <- a_k / sqrt(base_v_diag + 1e-12)

    # Step 3: Generate the final oscillator coefficients (NCP)
    u_sin_list[[j]] <- z_sin[[j]] * full_scale
    u_cos_list[[j]] <- z_cos[[j]] * full_scale
  }

  # --- 5. Synthesize Signal and Calculate Final Components ---
  S_t_matrix <- matrix(0, nrow = N, ncol = 3)
  Sigma_S_diag <- numeric(3) # Will store Var(S_j) for each band

  for (j in 1:3) {
    # Synthesize the j-th oscillator signal and mean-center it
    u_sin <- u_sin_list[[j]]
    u_cos <- u_cos_list[[j]]
    S_j <- sin_mat_list[[j]] %*% u_sin + cos_mat_list[[j]] %*% u_cos
    S_t_matrix[, j] <- S_j - mean(S_j)

    # Calculate its *exact* variance using the Gram matrices
    vj <- t(u_sin) %*% G_sin_list[[j]] %*% u_sin +
      t(u_cos) %*% G_cos_list[[j]] %*% u_cos +
      2 * t(u_sin) %*% G_sin_cos_list[[j]] %*% u_cos
    Sigma_S_diag[j] <- vj
  }

  # Calculate the time-varying scaling amplitude A(t)
  var_structured <- SDNN_t^2 * params$w
  denom_sq <- rowSums((p_j %*% diag(Sigma_S_diag)) * p_j)
  A_t <- sqrt(var_structured) / sqrt(denom_sq + 1e-12)

  # Combine components for the final mean trajectory
  sum_weighted_S <- rowSums(S_t_matrix * p_j)
  mu <- RR_t + A_t * sum_weighted_S

  # Generate the final noisy RRi signal
  var_noise <- SDNN_t^2 * (1 - params$w)
  RRi_t <- rnorm(N, mean = mu, sd = sqrt(var_noise))

  # --- 6. Format Output ---
  out <- data.table::data.table(
    t = t,
    RR = RRi_t,
    mu = mu,
    RR_baseline = RR_t,
    SDNN_t = SDNN_t,
    A_t = A_t,
    w = params$w,
    p_vlf = p_j[, 1],
    p_lf = p_j[, 2],
    p_hf = p_j[, 3]
  )

  return(list(
    data = out,
    freqs = freqs_list,
    log_v = log_v_list
  ))
}


#' Minimalistic function to get HRV frequency band power.
#'
#' Assumes best practices: spline interpolation, Hann window, and linear detrending.
#'
#' @param rr_ms A numeric vector of RR intervals in milliseconds.
#' @param bands A named list of frequency bands, each a c(min, max) vector in Hz.
#' @param fs The sampling rate in Hz for resampling. 4 Hz is standard.
#' @return A named numeric vector of the absolute power for each frequency band.

get_hrv_band_power <- function(rr_ms,
                               bands = list(
                                 vlf = c(0.003, 0.039),
                                 lf  = c(0.04, 0.149),
                                 hf  = c(0.15, 0.4)
                               ),
                               fs = 4) {

  # 1. Input validation
  stopifnot(
    is.numeric(rr_ms),
    length(rr_ms) > 10,
    all(rr_ms > 0 & !is.na(rr_ms))
  )

  # 2. Resample the RR series using cubic spline interpolation
  rr_times_s <- cumsum(rr_ms / 1000)
  interp_func <- stats::splinefun(rr_times_s, rr_ms, method = "natural")
  x <- interp_func(seq(rr_times_s[1], rr_times_s[length(rr_times_s)], by = 1 / fs))
  N <- length(x)

  # 3. Pre-processing: Detrend and apply a Hann window
  x <- stats::resid(stats::lm(x ~ seq_along(x))) # Detrend
  w <- 0.5 - 0.5 * cos(2 * pi * 0:(N - 1) / (N - 1)) # Hann window

  # 4. Calculate the one-sided Power Spectral Density (PSD)
  nfft <- 2^ceiling(log2(N))
  P <- (Mod(stats::fft(x * w))^2) / (fs * sum(w^2)) # Two-sided PSD
  P <- P[1:(nfft / 2 + 1)] * c(1, rep(2, nfft / 2 - 1), 1) # Convert to one-sided
  freqs <- (0:(nfft/2)) * (fs / nfft)
  df <- fs / nfft

  # 5. Calculate and return power for each band
  sapply(bands, function(band) {
    idx <- which(freqs >= band[1] & freqs < band[2])
    sum(P[idx]) * df
  })
}

#' Calculate moving proportions of HRV frequency band power.
#'
#' This wrapper function applies get_hrv_band_power over a sliding time window.
#' It can either calculate the time vector from the RR intervals or accept a
#' pre-calculated one.
#'
#' @param rr_ms A numeric vector of RR intervals in milliseconds.
#' @param rr_times_s (Optional) A numeric vector of timestamps in seconds
#'   corresponding to the start of each RR interval. Must be the same length
#'   as rr_ms. If NULL, it will be calculated automatically.
#' @param window_size_s The width of the sliding window in seconds. Default is 300s (5 mins).
#' @param step_size_s The amount the window moves forward, in seconds. Default is 60s.
#' @param bands A named list of frequency bands to analyze.
#' @param fs The sampling rate in Hz for resampling. 4 Hz is standard.
#' @return A data frame with the center time of each window and the proportional
#'   power for each band.

get_moving_hrv_proportions <- function(rr_ms,
                                       rr_times_s = NULL,
                                       window_size_s = 300,
                                       step_size_s = 60,
                                       bands = list(
                                         vlf = c(0.003, 0.039),
                                         lf  = c(0.040, 0.149),
                                         hf  = c(0.150, 0.400)
                                       ),
                                       fs = 4) {

  # 1. Validate inputs and establish the time vector
  stopifnot(is.numeric(rr_ms), length(rr_ms) > 20)

  if (is.null(rr_times_s)) {
    # If no time vector is provided, calculate it from the intervals
    rr_times_s <- cumsum(c(0, rr_ms[-length(rr_ms)])) / 1000
  } else {
    # If a time vector is provided, validate its length
    stopifnot(
      "Provided rr_times_s must have the same length as rr_ms" =
        length(rr_times_s) == length(rr_ms)
    )
  }

  # 2. Define the start times for each window
  window_starts <- seq(
    from = min(rr_times_s),
    to = max(rr_times_s) - window_size_s,
    by = step_size_s
  )

  # 3. Apply the analysis to each window
  results_list <- lapply(window_starts, function(start_t) {
    idx <- which(rr_times_s >= start_t & rr_times_s < (start_t + window_size_s))
    if (length(idx) < 20) return(NULL)

    abs_powers <- get_hrv_band_power(rr_ms[idx], bands, fs)
    total_power <- sum(abs_powers)

    if (total_power == 0) rep(0, length(bands)) else abs_powers / total_power
  })

  # 4. Combine results into a clean data frame
  results_df <- as.data.frame(do.call(rbind, results_list))
  # Add time column, filtering out any windows that were skipped (NULL)
  results_df$time_s <- window_starts[!sapply(results_list, is.null)] + (window_size_s / 2)

  results_df <- as.data.table(results_df)

  return(results_df)
}

#' Perform sliding-window analysis for time-domain metrics.
#'
#' @param data A tibble containing `time` and `RR_observed`.
#' @param window_sec The length of the sliding window in seconds.
#' @param overlap_perc The percentage of overlap between consecutive windows (0 to 1).
#' @param sampling_rate The sampling rate of the data in Hz.
#' @return A tibble with the window center time, and the calculated mean RR and SDNN.
perform_sliding_window_analysis <- function(data, window_sec, overlap_perc, sampling_rate) {
  window_samples <- window_sec * sampling_rate
  step_samples <- floor(window_samples * (1 - overlap_perc))

  # Calculate start indices for each window
  start_indices <- seq(1, nrow(data) - window_samples + 1, by = step_samples)

  results <- purrr::map_dfr(start_indices, ~{
    window_data <- data$RR[.x:(.x + window_samples - 1)]
    window_time <- data$t[.x + floor(window_samples / 2)] # Time at window center

    data.table(
      time = window_time,
      RR_windowed = mean(window_data, na.rm = TRUE),
      SDNN_windowed = sd(window_data, na.rm = TRUE)
    )
  })

  return(results)
}


#' Calculate multiple model fit metrics.
#'
#' This function computes several common metrics to evaluate the fit of a model,
#' including RMSE, R-squared, MAE, Bias, and MAPE.
#'
#' @param true_vals A numeric vector of ground-truth values.
#' @param estimated_vals A numeric vector of estimated values.
#' @return A tibble with RMSE, R2, MAE, Bias, and MAPE.
#'   MAPE will be NA if any true values are zero.
calculate_metrics <- function(true_vals, estimated_vals) {

  # Ensure no NA values interfere with calculations
  valid_indices <- !is.na(true_vals) & !is.na(estimated_vals)
  true <- true_vals[valid_indices]
  est  <- estimated_vals[valid_indices]

  if (length(true) == 0) {
    stop("No valid pairs of true and estimated values found.")
  }

  # --- Original Metrics ---
  # Root Mean Squared Error (RMSE)
  rmse <- sqrt(mean((true - est)^2))

  # R-squared (R²)
  r2   <- 1 - (sum((true - est)^2) / sum((true - mean(true))^2))

  # --- Added Metrics ---
  # Mean Absolute Error (MAE)
  mae  <- mean(abs(true - est))

  # Bias (or Mean Error, ME): Indicates if the model systematically over or underestimates.
  bias <- mean(est - true)

  # Mean Absolute Percentage Error (MAPE): Error as a percentage of true values.
  # This is undefined if any true values are zero.
  if (any(true == 0)) {
    warning("True values contain zeros. MAPE is returned as NA.")
    mape <- NA
  } else {
    mape <- mean(abs((true - est) / true)) * 100
  }

  # --- Format Output ---
  data.table::data.table(
    Metric = c("RMSE", "R2", "MAE", "Bias", "MAPE"),
    Value = c(rmse, r2, mae, bias, mape)
  )
}
