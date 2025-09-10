# Make sure you have the cmdstanr package installed and configured
# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# cmdstanr::install_cmdstan()

library(cmdstanr)
library(dplyr)
library(purrr)

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
find_mle_stan <- function(stan_model_file, stan_data, n_runs = 20, seed = 123) {

  # --- 1. Compile the Stan Model ---
  # The compilation is done only once, making the process efficient.
  cat("Compiling the Stan model...\n")
  model <- cmdstanr::cmdstan_model(stan_model_file)

  cat(paste0("Model compiled successfully. Starting ", n_runs, " optimization runs...\n"))

  # --- 2. Run the Optimizer Multiple Times ---
  # We use purrr::map to loop n_runs times. Each run uses a different
  # seed to ensure different random initial values.
  all_runs <- purrr::map(1:n_runs, function(i) {

    # Let Stan pick random initial values by not specifying the `init` argument.
    # The combination of the main seed and the per-run seed ensures reproducibility.
    fit <- model$optimize(
      data = stan_data,
      seed = seed + i # Increment seed for different random inits
    )

    # Return a list with the log-likelihood and the full fit object
    list(
      log_lik = fit$lp(),
      fit = fit
    )
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
  best_estimates <- best_fit$mle() %>%
    as_tibble(rownames = "parameter") %>%
    rename(estimate = value)

  cat(paste0(
    "Best run was #", best_index,
    " with a log-likelihood of: ", round(max(log_liks), 4), "\n"
  ))

  # Return a list with the key results
  return(
    list(
      best_run = best_fit,
      mle_estimates = best_estimates,
      log_lik = max(log_liks)
    )
  )
}


simulated_data[[1]]
