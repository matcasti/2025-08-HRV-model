REMOVE_FILES <- FALSE

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
  rm(files_to_remove, REMOVE_FILES)
}

closeAllConnections(); gc()

source("R/1-generate_data.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/2-classic_metrics.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/3-model-fitting.R"); rm(list = ls(all.names = TRUE)); gc()

## Needs to be run before script 4 for CWT data to be available
source("R/S9-cwt-comparator.R"); rm(list = ls(all.names = TRUE)); gc()

source("R/4-model_metrics.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/5-proof-of-concept.R"); rm(list = ls(all.names = TRUE)); gc()

source("R/S1-prior-predictive-checks.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S2-prior-sensitivity.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S2.1-quantification-of-approximation.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S2.2-pairwise-correlation.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S3-full-parameter-recovery.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S4-asymetric-dynamics.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S5-t-distribution.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S6-traceplots.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S7-model-diagnostics.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S8-ppchecks.R"); rm(list = ls(all.names = TRUE)); gc()
source("R/S10-window-sensitivity.R"); rm(list = ls(all.names = TRUE)); gc()

quarto::quarto_render("index.qmd")
quarto::quarto_render("supplementary.qmd")
