REMOVE_FILES <- TRUE

if(REMOVE_FILES) {
  files_to_remove <- c("data/error_statistics.RDS",
                       "data/error_stats_classic.RDS",
                       "data/model_predictions.RDS",
                       "models/model_fit_asymetric.RDS",
                       "models/model_t_dist.RDS",
                       "models/model_fit_poc.RDS",
                       "models/model_fit_prior_wide.RDS",
                       "models/model_fit_prior_narrow.RDS",
                       "models/model_fit_prior_normal.RDS",
                       "models/model_fit_scenario_1.RDS",
                       "models/model_fit_scenario_2.RDS",
                       "models/model_fit_scenario_3.RDS")
  unlink(files_to_remove)
}

source("R/1-generate_data.R"); gc()
source("R/2-classic_metrics.R"); gc()
source("R/3-model-fitting.R"); gc()
source("R/4-model_metrics.R"); gc()
source("R/5-proof-of-concept.R"); gc()

source("R/S1-prior-predictive-checks.R"); gc()
source("R/S2-prior-sensitivity.R"); gc()
source("R/S3-full-parameter-recovery.R"); gc()
source("R/S4-asymetric-dynamics.R"); gc()
source("R/S5-t-distribution.R"); gc()

quarto::quarto_render("index.qmd")
quarto::quarto_render("supplementary.qmd")
