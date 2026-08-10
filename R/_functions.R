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
#' @param ... Currently not used.
#' @param t_dist Logical. Use a t-distribution instead of a normal to generate the data. Default is FALSE.
#' @param nu Numeric. Nu parameter of the Student-t distribution controlling the tails of the distribution. Only used when `t_dist` is TRUE.
#'
#' @return A data frame containing the time vector 't', the final generated
#'   'RR' series, the underlying mean 'mu', and other key components,
#'   including p_vlf/p_lf/p_hf (the mixture/amplitude weights p_j(t); NOT
#'   power proportions) and q_vlf/q_lf/q_hf (the implied variance/power
#'   share q_j(t), comparable to conventional power-proportion estimates).
generate_rri_simulation <- function(N,
                                    t_max,
                                    params,
                                    N_sin,
                                    seed = 123,
                                    nu = NULL,
                                    t_dist = FALSE) {

  # Helper function to compute the GP's squared exponential covariance kernel
  gp_exp_quad_cov <- function(x, rho) {
    N <- length(x)
    K <- matrix(0, N, N)
    for (i in 1:N) {
      for (j in 1:N) {
        K[i, j] <- exp(-0.5 * ((x[i] - x[j]) / rho)^2)
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
  # Sign convention matches the main manuscript / Stan model exactly:
  # mu(t) = alpha + beta*D1(t) - c*beta*D2(t), so a perturbation-induced *drop*
  # requires a negative beta (see the params lists used throughout this project).
  RR_t <- params$alpha_r + params$beta_r * D_1(t) - params$c_r * params$beta_r * D_2(t)
  SDNN_t <- params$alpha_s + params$beta_s * D_1(t) - params$c_s * params$beta_s * D_2(t)
  C_t <- D_1(t) * (1 - params$c_c * D_2(t))

  # Time-varying band mixture/amplitude weights p_j(t) via convex
  # combination (NOT power proportions -- see q_j(t) below)
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

  # Scale log-frequencies to [0, 1] for a scale-agnostic GP
  log_freqs_scaled_list <- vector("list", 3)
  for (j in 1:3) {
    min_log_f <- min(log_freqs_list[[j]])
    range_log_f <- max(log_freqs_list[[j]]) - min_log_f
    log_freqs_scaled_list[[j]] <- (log_freqs_list[[j]] - min_log_f) / range_log_f
  }

  # Precompute sine/cosine basis matrices and Gram matrices
  sin_mat_list <- vector("list", 3)
  cos_mat_list <- vector("list", 3)
  G_sin_list <- vector("list", 3)
  G_cos_list <- vector("list", 3)
  G_sin_cos_list <- vector("list", 3)

  center_cols <- function(M) sweep(M, 2, colMeans(M), "-")

  for (j in 1:3) {
    T_mat <- outer(t * 60, freqs_list[[j]])
    # Center each basis column (sinusoid) BEFORE forming the Gram matrices.
    # This guarantees (a) S_j(t) = sin_mat %*% u_sin + cos_mat %*% u_cos is
    # automatically mean-zero for ANY coefficients, and (b) the Gram
    # matrices below are computed from the exact same centered basis that
    # generates the signal, so Var(S_j) is computed correctly.
    sin_mat_list[[j]] <- center_cols(sin(2 * pi * T_mat))
    cos_mat_list[[j]] <- center_cols(cos(2 * pi * T_mat))

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
    K <- gp_exp_quad_cov(log_freqs_scaled_list[[j]], params$rho_gp[j])
    K <- K + diag(1e-10, N_sin) # Scaled jitter
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
    # Synthesize the j-th oscillator signal using the mean-centered basis;
    # S_j is therefore exactly mean-centered by construction (matching the Stan
    # implementation's Step 5), and the Gram-matrix-based variance below is exact.
    u_sin <- u_sin_list[[j]]
    u_cos <- u_cos_list[[j]]
    S_j <- sin_mat_list[[j]] %*% u_sin + cos_mat_list[[j]] %*% u_cos
    S_t_matrix[, j] <- S_j - mean(S_j) # negligible residual mean kept as a safety check

    # Calculate its *exact* variance using the Gram matrices
    vj <- t(u_sin) %*% G_sin_list[[j]] %*% u_sin +
      t(u_cos) %*% G_cos_list[[j]] %*% u_cos +
      2 * t(u_sin) %*% G_sin_cos_list[[j]] %*% u_cos
    Sigma_S_diag[j] <- vj
  }

  # Calculate the time-varying scaling amplitude A(t)
  var_structured <- SDNN_t^2 * params$w
  variance_terms <- (p_j %*% diag(Sigma_S_diag)) * p_j
  denom_sq <- rowSums(variance_terms)
  A_t <- sqrt(var_structured) / sqrt(denom_sq + 1e-12)

  # Implied variance (power) share per band: q_j(t).
  # NOTE: p_j(t) are mixture/amplitude weights, NOT power proportions (see
  # manuscript). q_j(t) is the actual proportion of the structured signal's
  # variance attributable to band j at time t, and is the correct
  # ground-truth target when validating against power-based estimators
  # (e.g., STFT/windowed periodogram methods).
  q_j <- variance_terms / (denom_sq + 1e-12)
  colnames(q_j) <- c("VLF", "LF", "HF")

  # Combine components for the final mean trajectory
  sum_weighted_S <- rowSums(S_t_matrix * p_j)
  mu <- RR_t + A_t * sum_weighted_S

  # Generate the final noisy RRi signal
  var_noise <- SDNN_t^2 * (1 - params$w)

  if (t_dist) {
    sigma_t <- sqrt(var_noise * (nu - 2) / nu)
    # Draw from standardized Student-t and scale/shift
    RRi_t <- mu + sigma_t * rt(N, df = nu)
  } else {
    RRi_t <- rnorm(N, mean = mu, sd = sqrt(var_noise))
  }

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
    p_hf = p_j[, 3],
    q_vlf = q_j[, 1],
    q_lf = q_j[, 2],
    q_hf = q_j[, 3]
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
  # Guard against step_samples flooring to 0 for short windows combined with
  # very high overlap (e.g., window_sec=30, sampling_rate=2, overlap_perc=0.99
  # gives floor(60 * 0.01) = 0), which would make the seq() below fail with an
  # invalid (zero) step size.
  step_samples <- max(1, floor(window_samples * (1 - overlap_perc)))

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

#' Save a data.frame/data.table as a Pandoc/GitHub-style markdown pipe table.
#'
#' Thin wrapper around knitr::kable(format = "pipe") so every table-generating
#' script in this project exports a copy that can be pasted directly into a
#' `[TABLE IN MARKDOWN HERE]` placeholder in the .qmd manuscripts, alongside
#' the existing `gt`-object RDS export used for the polished PDF/Word pipeline.
#'
#' @param dt A data.frame or data.table to export.
#' @param path Output file path (e.g., "tables/tbl-1.md").
#' @param ... Additional arguments passed to knitr::kable() (e.g., align, col.names).
#' @return Invisibly, the markdown character vector that was written.
save_markdown_table <- function(dt, path, ...) {
  md <- knitr::kable(dt, format = "pipe", ...)
  writeLines(md, con = path)
  invisible(md)
}

#' Run the conventional (sliding-window + STFT) analysis and benchmark it
#' against ground truth, for a single simulated scenario.
#'
#' This is a refactor of the per-scenario loop body originally in
#' 2-classic_metrics.R, parameterized by window length so it can be reused
#' for the window-length sensitivity analysis (Supplementary Section S4.3.1)
#' without duplicating the alignment/metric logic.
#'
#' @param sim_data_i A data.table as produced by generate_rri_simulation()$data
#'   (must contain t, RR, RR_baseline, SDNN_t, q_vlf, q_lf, q_hf).
#' @param window_seconds Sliding-window / STFT window length, in seconds.
#' @param overlap_perc Fractional overlap for the time-domain sliding window.
#' @param sampling_rate_hz Resampling rate used internally by the STFT step.
#' @return A list with `estimates` (the full aligned comparison data.table)
#'   and `statistics` (a named list of calculate_metrics() outputs).
run_classic_comparison <- function(sim_data_i, window_seconds = 60,
                                   overlap_perc = 0.99, sampling_rate_hz = 2) {
  time_domain_results <- perform_sliding_window_analysis(
    data = sim_data_i,
    window_sec = window_seconds,
    overlap_perc = overlap_perc,
    sampling_rate = sampling_rate_hz
  )

  spectral_results <- get_moving_hrv_proportions(
    rr_ms = sim_data_i$RR,
    rr_times_s = sim_data_i$t * 60,
    window_size_s = window_seconds,
    step_size_s = 1
  )

  aligned_results <- data.table::data.table(
    t = sim_data_i$t,
    RR_windowed_interp = approx(time_domain_results$time, time_domain_results$RR_windowed, xout = sim_data_i$t, rule = 2)$y,
    SDNN_windowed_interp = approx(time_domain_results$time, time_domain_results$SDNN_windowed, xout = sim_data_i$t, rule = 2)$y,
    p_vlf_stft_interp = approx(spectral_results$time / 60, spectral_results$vlf, xout = sim_data_i$t, rule = 2)$y,
    p_lf_stft_interp = approx(spectral_results$time / 60, spectral_results$lf, xout = sim_data_i$t, rule = 2)$y,
    p_hf_stft_interp = approx(spectral_results$time / 60, spectral_results$hf, xout = sim_data_i$t, rule = 2)$y
  )

  full_comparison_data <- sim_data_i[aligned_results, on = "t"]

  # NOTE: STFT estimates a power proportion, so it is validated against
  # q_j(t) (implied variance/power share), not p_j(t) (mixture weights).
  statistics <- list(
    rr_metrics = calculate_metrics(full_comparison_data$RR_baseline, full_comparison_data$RR_windowed_interp),
    sdnn_metrics = calculate_metrics(full_comparison_data$SDNN_t, full_comparison_data$SDNN_windowed_interp),
    vlf_metrics = calculate_metrics(full_comparison_data$q_vlf, full_comparison_data$p_vlf_stft_interp),
    lf_metrics = calculate_metrics(full_comparison_data$q_lf, full_comparison_data$p_lf_stft_interp),
    hf_metrics = calculate_metrics(full_comparison_data$q_hf, full_comparison_data$p_hf_stft_interp)
  )

  list(estimates = full_comparison_data, statistics = statistics)
}

#' Block-bootstrap confidence intervals for calculate_metrics() outputs.
#'
#' The windowed, STFT, and CWT comparators are deterministic given the data
#' and so have no native posterior uncertainty (unlike the generative model).
#' To report comparable uncertainty in Table 3, we resample contiguous BLOCKS
#' of (true, estimated) pairs with replacement (a moving block bootstrap;
#' Kunsch 1989) rather than resampling individual time points, since a time
#' series' autocorrelation would make an i.i.d. bootstrap overconfident
#' (understate the true sampling variability).
#'
#' @param true_vals Numeric vector of ground-truth values, ordered in time.
#' @param estimated_vals Numeric vector of estimates, same order/length.
#' @param n_boot Number of bootstrap replicates.
#' @param block_len Block length in observations. Defaults to the standard
#'   rule-of-thumb round(2.5 * n^(1/3)); override for a specific sensitivity
#'   check.
#' @param seed Optional seed for reproducibility.
#' @return A data.table with one row per Metric: the point estimate (from the
#'   full, unresampled data) plus a 95% bootstrap CI (CI_low, CI_high).
block_bootstrap_metrics <- function(true_vals, estimated_vals, n_boot = 2000,
                                    block_len = NULL, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  valid <- !is.na(true_vals) & !is.na(estimated_vals)
  true_vals <- true_vals[valid]
  estimated_vals <- estimated_vals[valid]
  n <- length(true_vals)

  if (is.null(block_len)) {
    block_len <- max(1, round(2.5 * n^(1 / 3)))
  }
  block_len <- min(block_len, n)

  point_estimate <- calculate_metrics(true_vals, estimated_vals)

  n_blocks_needed <- ceiling(n / block_len)
  max_start <- n - block_len + 1

  boot_results <- vector("list", n_boot)
  for (b in seq_len(n_boot)) {
    block_starts <- sample.int(max_start, size = n_blocks_needed, replace = TRUE)
    idx <- unlist(lapply(block_starts, function(s) s:(s + block_len - 1)))
    idx <- idx[seq_len(n)]
    boot_results[[b]] <- calculate_metrics(true_vals[idx], estimated_vals[idx])
  }

  boot_dt <- data.table::rbindlist(boot_results, idcol = "rep")

  ci <- boot_dt[, list(
    CI_low = stats::quantile(Value, 0.025, na.rm = TRUE, names = FALSE),
    CI_high = stats::quantile(Value, 0.975, na.rm = TRUE, names = FALSE)
  ), keyby = Metric]

  merge(point_estimate, ci, by = "Metric", sort = FALSE)
}

#' Continuous Morlet wavelet transform (Torrence & Compo 1998), evaluated at
#' a fixed set of center frequencies via FFT-based convolution.
#'
#' Implemented directly (rather than depending on an external wavelet
#' package) so the exact normalization is known and testable. Validated
#' against known-frequency test signals: single-tone frequency localization,
#' and multi-tone equal-amplitude band-integrated power recovery (see project
#' notes) -- for the WIDE, canonical VLF/LF/HF band widths used in this
#' analysis, the raw (uncorrected) |W(s)|^2 power, integrated over each band,
#' recovers equal-amplitude tone proportions to within ~0.3%. The classic
#' "global wavelet spectrum" 1/scale bias correction (appropriate for
#' comparing POINT spectral density across scales, e.g., against a Fourier
#' periodogram) is deliberately NOT applied here, because it over-corrects
#' when power is integrated across bands much wider than the wavelet's own
#' bandwidth at that scale, which is the situation for canonical HRV bands.
#'
#' @param x Numeric vector, evenly sampled.
#' @param fs Sampling rate in Hz.
#' @param freqs Numeric vector of center frequencies (Hz) at which to
#'   evaluate wavelet power.
#' @param w0 Morlet non-dimensional frequency parameter (default 6, the
#'   standard choice balancing time and frequency resolution).
#' @return A list with `power` (length(x) x length(freqs) matrix of raw
#'   wavelet power), `freqs`, and `scales` (the Fourier-equivalent scale used
#'   for each frequency).
morlet_cwt <- function(x, fs, freqs, w0 = 6) {
  n <- length(x)
  dt <- 1 / fs
  n_pad <- 2^ceiling(log2(n))
  x_padded <- c(x, rep(0, n_pad - n))
  x_fft <- stats::fft(x_padded)

  k <- 0:(n_pad - 1)
  ang_freq <- 2 * pi * k / (n_pad * dt)
  neg_idx <- k > n_pad / 2
  ang_freq[neg_idx] <- -2 * pi * (n_pad - k[neg_idx]) / (n_pad * dt)

  # Fourier-equivalent scale for each target frequency (Torrence & Compo
  # 1998, Table 1, for the Morlet wavelet).
  scales <- (w0 + sqrt(2 + w0^2)) / (4 * pi * freqs)

  power <- matrix(NA_real_, nrow = n, ncol = length(freqs))
  for (j in seq_along(scales)) {
    s <- scales[j]
    # Normalized Morlet wavelet in the frequency domain (unit energy at
    # every scale, so different scales are directly comparable).
    psi_hat <- sqrt(2 * pi * s / dt) * (pi^-0.25) *
      exp(-((s * ang_freq - w0)^2) / 2) * (ang_freq > 0)
    W <- stats::fft(x_fft * psi_hat, inverse = TRUE) / n_pad
    power[, j] <- (Mod(W[1:n]))^2
  }

  list(power = power, freqs = freqs, scales = scales)
}

#' Time-resolved VLF/LF/HF power proportions via continuous (Morlet) wavelet
#' transform, mirroring get_moving_hrv_proportions()'s interface and output
#' structure so it is a drop-in comparator alongside the STFT-based one.
#'
#' Unlike the windowed STFT estimator, the CWT provides a continuous-time
#' estimate at every resampled time point (no window/step parameters are
#' needed) -- band power is integrated across each canonical band's frequency
#' range at each time point, then normalized so the three bands sum to 1 at
#' that time point (the same per-time-step normalization convention used for
#' the STFT comparator). Edge time points are subject to some wavelet
#' "cone of influence" boundary distortion, a known CWT limitation; this is
#' not separately masked here, consistent with how the STFT comparator's
#' edge windows are handled (flat extrapolation via approx(..., rule = 2)
#' downstream).
#'
#' Detrending: unlike get_hrv_band_power() (called independently within each
#' short STFT window, so a simple linear detrend is adequate), the CWT
#' operates on the full record at once. A single global linear detrend
#' leaves most of a non-linear (e.g., double-logistic perturbation-recovery)
#' trend's energy in the residual, which is then misattributed to the VLF
#' band. A moving-average highpass filter (window = highpass_window_s) is
#' used instead; the default (60s, matching the primary comparator window
#' elsewhere in this project) was chosen by checking that it drives the
#' residual's correlation with the known ground-truth trend to ~0 on
#' simulated data with a known trend, closely matching the best achievable
#' (oracle-detrended) residual variance.
#'
#' @param rr_ms A numeric vector of RR intervals in milliseconds.
#' @param rr_times_s (Optional) timestamps in seconds for each rr_ms
#'   observation. If NULL, computed from cumulative rr_ms.
#' @param bands A named list of frequency bands (Hz).
#' @param fs Resampling rate in Hz (matches get_hrv_band_power's convention).
#' @param n_freq Number of frequencies to scan across the full band range
#'   (log-spaced), concatenated; higher values give a finer scalogram at
#'   increased compute cost.
#' @param highpass_window_s Moving-average highpass window, in seconds, used
#'   to remove slow trend before the wavelet transform.
#' @return A data.table with `time_s` and one column per band, giving the
#'   normalized (proportional) power at each time point.
get_cwt_band_proportions <- function(rr_ms,
                                     rr_times_s = NULL,
                                     bands = list(
                                       vlf = c(0.003, 0.039),
                                       lf  = c(0.040, 0.149),
                                       hf  = c(0.150, 0.400)
                                     ),
                                     fs = 4,
                                     n_freq = 120,
                                     highpass_window_s = 60) {
  stopifnot(is.numeric(rr_ms), length(rr_ms) > 20)

  if (is.null(rr_times_s)) {
    rr_times_s <- cumsum(c(0, rr_ms[-length(rr_ms)])) / 1000
  }

  # Resample onto a regular grid via cubic spline interpolation (matches
  # get_hrv_band_power's convention).
  interp_func <- stats::splinefun(rr_times_s, rr_ms, method = "natural")
  time_grid <- seq(rr_times_s[1], rr_times_s[length(rr_times_s)], by = 1 / fs)
  x_raw <- interp_func(time_grid)

  # Moving-average highpass: subtract a centered moving average, with
  # edge samples filled by carrying the nearest valid average forward/back.
  win_n <- round(highpass_window_s * fs)
  if (win_n %% 2 == 0) win_n <- win_n + 1
  win_n <- min(win_n, length(x_raw) - 1)
  moving_avg <- stats::filter(x_raw, rep(1 / win_n, win_n), sides = 2)
  moving_avg <- zoo::na.locf(zoo::na.locf(moving_avg, na.rm = FALSE), fromLast = TRUE)
  x <- x_raw - as.numeric(moving_avg)

  band_range <- range(unlist(bands))
  # Linearly spaced scan frequencies -- matches what was numerically
  # validated (near-exact recovery of equal-amplitude multi-tone band
  # proportions). Log-spacing was tried first but allocates disproportionately
  # more scan points to the VLF band (which spans a much wider log-range than
  # LF/HF), double-counting overlapping wavelet responses there and biasing
  # the integrated VLF power upward.
  freqs_scan <- seq(band_range[1], band_range[2], length.out = n_freq)

  cwt <- morlet_cwt(x, fs = fs, freqs = freqs_scan)

  band_power <- sapply(bands, function(b) {
    idx <- which(freqs_scan >= b[1] & freqs_scan <= b[2])
    rowSums(cwt$power[, idx, drop = FALSE])
  })

  total_power <- rowSums(band_power)
  band_prop <- band_power / ifelse(total_power == 0, NA, total_power)

  out <- data.table::as.data.table(band_prop)
  out[, time_s := time_grid]
  out[]
}

#' Run the CWT-based analysis and benchmark it against ground truth, for a
#' single simulated scenario. Structurally mirrors run_classic_comparison()
#' so it plugs into the same Table 3 assembly pipeline.
#'
#' @param sim_data_i A data.table as produced by generate_rri_simulation()$data.
#' @param fs Resampling rate in Hz passed to get_cwt_band_proportions().
#' @param n_freq Number of scan frequencies passed to get_cwt_band_proportions().
#' @return A list with `estimates` and `statistics`, matching
#'   run_classic_comparison()'s return structure.
run_cwt_comparison <- function(sim_data_i, fs = 4, n_freq = 120) {
  cwt_results <- get_cwt_band_proportions(
    rr_ms = sim_data_i$RR,
    rr_times_s = sim_data_i$t * 60,
    fs = fs,
    n_freq = n_freq
  )

  aligned_results <- data.table::data.table(
    t = sim_data_i$t,
    p_vlf_cwt_interp = approx(cwt_results$time_s / 60, cwt_results$vlf, xout = sim_data_i$t, rule = 2)$y,
    p_lf_cwt_interp = approx(cwt_results$time_s / 60, cwt_results$lf, xout = sim_data_i$t, rule = 2)$y,
    p_hf_cwt_interp = approx(cwt_results$time_s / 60, cwt_results$hf, xout = sim_data_i$t, rule = 2)$y
  )

  full_comparison_data <- sim_data_i[aligned_results, on = "t"]

  statistics <- list(
    vlf_metrics = calculate_metrics(full_comparison_data$q_vlf, full_comparison_data$p_vlf_cwt_interp),
    lf_metrics = calculate_metrics(full_comparison_data$q_lf, full_comparison_data$p_lf_cwt_interp),
    hf_metrics = calculate_metrics(full_comparison_data$q_hf, full_comparison_data$p_hf_cwt_interp)
  )

  list(estimates = full_comparison_data, statistics = statistics)
}
