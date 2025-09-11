# Make sure you have the cmdstanr package installed and configured
# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# cmdstanr::install_cmdstan()

library(cmdstanr)

#' Find the Maximum Likelihood Estimate using Multi-Start Optimization
#'
#' This function compiles a Stan model and runs its optimizer multiple times from
#' random starting points to robustly search for the global Maximum Likelihood
#' Estimate (MLE).
#'
#' @param stan_model_file A string containing the path to the .stan model file.
#'   The model block should have priors commented out to find the MLE.
#' @param stan_data A list containing the data for the Stan model.
#' @param n_runs An integer specifying the number of optimization runs to perform.
#'   More runs increase the chance of finding the global maximum.
#' @param seed An integer for the random number generator to ensure reproducibility.
#'
#' @return A list containing:
#'   - `best_run`: An object containing the full results from the best run.
#'   - `mle_estimates`: A tibble/data.frame with the parameter estimates from the best run.
#'   - `log_lik`: The maximum log-likelihood value found.
#'
#' @examples
#' \dontrun{
#' # Assuming 'model_mle.stan' is your model with priors commented out
#' # and 'my_data' is a list with N, t, RR, etc.
#'
#' mle_results <- find_mle_stan("model_mle.stan", my_data, n_runs = 50)
#'
#' # View the best parameter estimates
#' print(mle_results$mle_estimates)
#'
#' # View the maximum log-likelihood
#' print(mle_results$log_lik)
#' }
find_mle_stan <- function(stan_model_file, time, RR, N_sin = 25, n_runs = 20, iter = 1000, seed = 123) {

  # --- 1. Compile the Stan Model ---
  # The compilation is done only once, making the process efficient.
  cat("Compiling the Stan model...\n")
  model <- cmdstanr::cmdstan_model(stan_model_file)

  cat(paste0("Model compiled successfully. Starting ", n_runs, " optimization runs...\n"))

  stan_data <- list(
    N = length(time),
    t = time,
    RR = RR,
    N_sin = N_sin,
    freqs = list(
      seq(0.003, 0.039, length.out = N_sin),
      seq(0.04, 0.149, length.out = N_sin),
      seq(0.15, 0.4, length.out = N_sin)
    )
  )

  # --- 2. Run the Optimizer Multiple Times ---
  # We use purrr::map to loop n_runs times. Each run uses a different
  # seed to ensure different random initial values.
  all_runs <- purrr::map(1:n_runs, function(i) {
    # Let Stan pick random initial values by not specifying the `init` argument.
    # The combination of the main seed and the per-run seed ensures reproducibility.
    suppressWarnings({
      fit <- model$optimize(
        data = stan_data,
        seed = seed + i, # Increment seed for different random inits
        iter = iter,
        show_messages = FALSE,
        show_exceptions = FALSE,
        refresh = 0
      )
    })

    # Return a list with the log-likelihood and the full fit object
    if (fit$return_codes() == 0) {
      list(log_lik = fit$lp(), fit = fit)
    } else {
      list(log_lik = NA, fit = NA)
    }
  }, .progress = TRUE) # Show a progress bar

  # --- 3. Find and Return the Best Run ---
  cat("All runs complete. Finding the best result...\n")

  # Extract the log-likelihood from each run
  log_liks <- purrr::map_dbl(all_runs, "log_lik")

  # Find the index of the run with the maximum log-likelihood
  best_index <- which.max(log_liks)

  # Get the best fit object
  best_fit <- all_runs[[best_index]]$fit

  # Extract the parameter estimates from the best run
  best_estimates <- best_fit$mle() |>
    as.data.table(keep.rownames = "parameter")

  message(paste0(
    "Best run was #", best_index,
    " with a log-likelihood of: ", round(log_liks[best_index], 4), "\n"
  ))

  mle_estimates <- best_estimates[
    i = V1 %like% "tau|delta|phi|lambda|alpha|beta|^c|^pi_|^w|^b" &
      !V1 %like% "log$|logit$",
    j = list(Parameter = V1, Estimate = V2)
  ]

  predicted_rri <- data.table::data.table(
    time = time,
    mu = best_estimates[V1 %like% "^mu", V2],
    RR_baseline = best_estimates[V1 %like% "^RR_baseline", V2],
    SDNN_t = best_estimates[V1 %like% "^SDNN_t", V2],
    p_vlf = best_estimates[V1 %like% "^p_t" & V1 %like% "1\\]$", V2],
    p_lf = best_estimates[V1 %like% "^p_t" & V1 %like% "2\\]$", V2],
    p_hf = best_estimates[V1 %like% "^p_t" & V1 %like% "3\\]$", V2],
    C_t = best_estimates[V1 %like% "^C_t", V2]
  )

  # Return a list with the key results
  return(
    list(
      mle_estimates = mle_estimates,
      log_lik = log_liks[best_index],
      predicted_rri = predicted_rri
    )
  )
}

library(data.table)

rr_data <- CardioCurveR::import_RRi_txt(
  file = "../../Desarrollo web/Polar H10 R-R interval/R/polar_h10_rr_only_mi-pollito_2025-08-28_01-19-14.txt"
) |> as.data.table()

mle_estimate <-
  find_mle_stan(stan_model_file = "models/rri_mle_model.stan",
                time = rr_data$time,
                RR = rr_data$RRi,
                N_sin = 10,
                n_runs = 100,
                iter = 1000,
                seed = 123)

mle_estimate$mle_estimates[, round(Estimate, 3), Parameter]

mle_estimate$predicted_rri[, plot(time, mu, type = "l", ylim = c())]
rr_data[, lines(time, RRi, col = "red")]
