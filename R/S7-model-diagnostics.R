
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(rstan)
library(gt)

## Load model fitted
model <- readRDS("models/model_fit_prior_normal.RDS")

posterior <- as.data.table(model, pars = c(
  "lambda","phi","tau","delta",
  "alpha_r","beta_r","c_r",
  "alpha_s","beta_s","c_s",
  "c_c", "w", "pi_base", "pi_pert"
))

posterior[, row_id := seq_len(.N)]

posterior_long <- melt.data.table(posterior,
                                  id.vars = "row_id",
                                  variable.name = "Parameter")

levels(posterior_long$Parameter) <-
  c("$\\lambda$", "$\\phi$", "$\\tau$", "$\\delta$",
    "$\\alpha_r$", "$\\beta_r$", "$c_r$",
    "$\\alpha_s$", "$\\beta_s$", "$c_s$",
    "$c_c$", "$w$",
    "$\\vec{\\pi}_{base}$ [VLF]", "$\\vec{\\pi}_{base}$ [LF]", "$\\vec{\\pi}_{base}$ [HF]",
    "$\\vec{\\pi}_{pert}$ [VLF]", "$\\vec{\\pi}_{pert}$ [LF]", "$\\vec{\\pi}_{pert}$ [HF]")

diagnostic_tbl <- posterior_long[, list(
  `MCSE Mean` = round(posterior::mcse_mean(value), 3),
  `Bulk ESS` = posterior::ess_bulk(value),
  `Tail ESS` = posterior::ess_tail(value),
  `R-hat` = round(posterior::rhat(value), 5)
), keyby = Parameter]

gt(diagnostic_tbl) |>
  fmt_markdown(columns = "Parameter") |>
  opt_stylize(style = 5) |>
  tab_style(style = cell_text(size = "small"),
            locations = list(cells_body(),
                             cells_column_labels(),
                             cells_column_spanners())) |>
  saveRDS(file = "tables/tbl-S1.RDS")
