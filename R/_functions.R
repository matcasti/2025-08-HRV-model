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
                                 vlf = c(0.003, 0.04),
                                 lf  = c(0.04, 0.15),
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
    window_data <- data$RR_observed[.x:(.x + window_samples - 1)]
    window_time <- data$time[.x + floor(window_samples / 2)] # Time at window center

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
  # Ensure tibble is available for the output format
  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("Package 'tibble' is needed for this function. Please install it.", call. = FALSE)
  }

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
  tibble::tibble(
    Metric = c("RMSE", "R2", "MAE", "Bias", "MAPE"),
    Value = c(rmse, r2, mae, bias, mape)
  )
}
