
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(rstan)
library(tidybayes)
library(correlation)

pars <- c(
  "lambda","phi","tau","delta",
  "alpha_r","beta_r","c_r",
  "alpha_s","beta_s","c_s",
  "c_c"
)

col_names <- c(
  "$\\lambda$","$\\phi$","$\\tau$","$\\delta$",
  "$\\alpha_r$","$\\beta_r$","$c_r$",
  "$\\alpha_s$","$\\beta_s$","$c_s$",
  "$c_c$"
)

## Load models
fit_poc <- readRDS("models/model_fit_poc.RDS") |> extract(pars = pars) |> as.data.table()
fit_scenario_1 <- readRDS("models/model_fit_scenario_1.RDS") |> extract(pars = pars) |> as.data.table()
fit_scenario_2 <- readRDS("models/model_fit_scenario_2.RDS") |> extract(pars = pars) |> as.data.table()
fit_scenario_3 <- readRDS("models/model_fit_scenario_3.RDS") |> extract(pars = pars) |> as.data.table()

names(fit_poc) <-
  names(fit_scenario_1) <-
  names(fit_scenario_2) <-
  names(fit_scenario_3) <-
  col_names

# -------------------------------------------------------------------------

corr_results <- list(
  `Empirical case` = correlation(fit_poc, method = "gaussian") |> as.data.table(),
  `Scenario (1)` = correlation(fit_scenario_1, method = "gaussian") |> as.data.table(),
  `Scenario (2)` = correlation(fit_scenario_2, method = "gaussian") |> as.data.table(),
  `Scenario (3)` = correlation(fit_scenario_3, method = "gaussian") |> as.data.table()
) |> rbindlist(idcol = "Model")

corr_results <- corr_results[,list(Parameter = paste0(Parameter1, " ~ ", Parameter2),
                                   Estimate = paste0(round(r, 2), " [", round(CI_low, 2),", ", round(CI_high, 2),"]")),
                             keyby = list(Model)]

corr_results <- dcast.data.table(corr_results, Parameter ~ Model, value.var = "Estimate")

knitr::kable(corr_results, align = "l") |>
  saveRDS("tables/tbl-S2.RDS")
