
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
  "c_c", "w", "pi_base", "pi_pert",
  "rho_gp"
))

posterior[, row_id := seq_len(.N)]

posterior_long <- melt.data.table(posterior,
                                  id.vars = "row_id",
                                  variable.name = "Parameter")

# Map each parameter's actual (rstan-generated) column name to its display
# label, rather than relabeling by position -- this stays correct
# regardless of the declaration order in the .stan file.
label_map <- c(
  lambda = "$\\lambda$", phi = "$\\phi$", tau = "$\\tau$", delta = "$\\delta$",
  alpha_r = "$\\alpha_r$", beta_r = "$\\beta_r$", c_r = "$c_r$",
  alpha_s = "$\\alpha_s$", beta_s = "$\\beta_s$", c_s = "$c_s$",
  c_c = "$c_c$", w = "$w$",
  `pi_base[1]` = "$\\vec{\\pi}_{base}$ [VLF]", `pi_base[2]` = "$\\vec{\\pi}_{base}$ [LF]", `pi_base[3]` = "$\\vec{\\pi}_{base}$ [HF]",
  `pi_pert[1]` = "$\\vec{\\pi}_{pert}$ [VLF]", `pi_pert[2]` = "$\\vec{\\pi}_{pert}$ [LF]", `pi_pert[3]` = "$\\vec{\\pi}_{pert}$ [HF]",
  `rho_gp[1]` = "$\\rho_{gp}$ [VLF]", `rho_gp[2]` = "$\\rho_{gp}$ [LF]", `rho_gp[3]` = "$\\rho_{gp}$ [HF]"
)

posterior_long[, Parameter := factor(as.character(Parameter),
                                     levels = names(label_map),
                                     labels = unname(label_map))]

diagnostic_tbl <- posterior_long[, list(
  `MCSE Mean` = round(posterior::mcse_mean(value), 3),
  `Bulk ESS` = posterior::ess_bulk(value),
  `Tail ESS` = posterior::ess_tail(value),
  `R-hat` = round(posterior::rhat(value), 5)
), keyby = Parameter]

# Non-centered deviates (z_gp, z_sin, z_cos): one element per sinusoid per
# band. Per Section S1.3, report the minimum, median, and maximum R-hat and
# ESS across the full set of individual deviates for each of z_gp/z_sin/z_cos
# (three summary rows per type), rather than per-element (which would add
# dozens of rows) or a single collapsed worst-case value (which the text
# does not ask for).
posterior_z <- as.data.table(model, pars = c("z_gp", "z_sin", "z_cos"))
posterior_z[, row_id := seq_len(.N)]

posterior_z_long <- melt.data.table(posterior_z,
                                    id.vars = "row_id",
                                    variable.name = "Parameter")
posterior_z_long[, Group := sub("\\[.*\\]$", "", Parameter)]

posterior_z_diag <- posterior_z_long[, list(
  mcse_mean = posterior::mcse_mean(value),
  ess_bulk = posterior::ess_bulk(value),
  ess_tail = posterior::ess_tail(value),
  rhat = posterior::rhat(value)
), keyby = list(Group, Parameter)]

diagnostic_tbl_z <- posterior_z_diag[, list(
  Statistic = c("Min", "Median", "Max"),
  `MCSE Mean` = round(c(min(mcse_mean), median(mcse_mean), max(mcse_mean)), 3),
  `Bulk ESS` = round(c(min(ess_bulk), median(ess_bulk), max(ess_bulk))),
  `Tail ESS` = round(c(min(ess_tail), median(ess_tail), max(ess_tail))),
  `R-hat` = round(c(min(rhat), median(rhat), max(rhat)), 5)
), keyby = Group]

diagnostic_tbl_z[, Parameter := fcase(
  Group == "z_gp", paste0("$z_{gp}$ (", Statistic, ")"),
  Group == "z_sin", paste0("$z_{sin}$ (", Statistic, ")"),
  Group == "z_cos", paste0("$z_{cos}$ (", Statistic, ")")
)]
diagnostic_tbl_z[, c("Group", "Statistic") := NULL]
setcolorder(diagnostic_tbl_z, c("Parameter", "MCSE Mean", "Bulk ESS", "Tail ESS", "R-hat"))

diagnostic_tbl <- rbind(diagnostic_tbl, diagnostic_tbl_z)

knitr::kable(diagnostic_tbl, align = "l") |>
  saveRDS(file = "tables/tbl-S3.RDS")
