REMOVE_FILES <- TRUE

if(REMOVE_FILES) {
  files_to_remove <- c("data/error_statistics.RDS",
                       "data/error_stats_classic.RDS",
                       "data/model_predictions.RDS",
                       "models/model_fit_poc.RDS",
                       "models/model_fit_prior_wide.RDS",
                       "models/model_fit_prior_narrow.RDS",
                       "models/model_fit_prior_normal.RDS",
                       "models/model_fit_scenario_1.RDS",
                       "models/model_fit_scenario_2.RDS",
                       "models/model_fit_scenario_3.RDS")
  unlink(files_to_remove)
}

source("R/1-generate_data.R")
source("R/2-classic_metrics.R")
source("R/3-model-fitting.R")
source("R/4-model_metrics.R")
source("R/5-proof-of-concept.R")

source("R/S1-prior-predictive-checks.R")
source("R/S2-prior-sensitivity.R")

quarto::quarto_render("index.qmd")
quarto::quarto_render("supplementary.qmd")
