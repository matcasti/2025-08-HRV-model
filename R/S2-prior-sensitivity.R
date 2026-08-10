
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(CardioCurveR)
library(rstan)
library(ggplot2)

## Import functions
source("R/_functions.R")

## Load the data
poc_data <- import_RRi_txt(file = "data-raw/rri-jabf.txt",
                           remove_ectopic = TRUE,
                           filter_noise = FALSE) |>
  as.data.table()

## Visualize the data
plot(poc_data, type = "l", col = "gray")
points(poc_data, pch = 16, cex = 0.5); grid()

# Prepare data ------------------------------------------------------------

## Get an initial estimation of logistic parameters
prior_params <- with(poc_data, estimate_RRi_curve(time, RRi))$parameters |> abs()

N_sin <- 50

## Stan data
stan_data <- list(
  N = length(poc_data$time),
  t = poc_data$time,
  RR = poc_data$RRi,
  N_sin = N_sin,
  freqs = list(
    seq(0.003, 0.039, length.out = N_sin), # VLF
    seq(0.040, 0.149, length.out = N_sin), # LF
    seq(0.150, 0.400, length.out = N_sin)  # HF
  ),
  lambda_mu = prior_params[["lambda"]],
  phi_mu = prior_params[["phi"]],
  tau_mu = prior_params[["tau"]],
  delta_mu = prior_params[["delta"]]
)

## Precompile the model
model <- stan_model(file = "models/prior_custom_rri_model.stan")

## Specify the width of the priors
prior_mult <- c(narrow = 1/2, normal = 1, wide = 2)

# Model fitting process ---------------------------------------------------

for (i in 1:3) {
  stan_data$prior_mult <- prior_mult[[i]]
  prior_name <- names(prior_mult)[[i]]
  # Fit the model -----------------------------------------------------------
  model_fit <- rstan::sampling(
    object = model,
    init = 0,
    pars = c(
      "lambda","phi","tau","delta",
      "alpha_r","beta_r","c_r",
      "alpha_s","beta_s","c_s",
      "c_c", "w", "pi_base", "pi_pert",
      "rho_gp", "z_gp", "z_sin", "z_cos"
    ),
    data = stan_data,
    iter = 10000, warmup = 5000,
    chains = 4, cores = 4,
    seed = 1234,
    control = list(adapt_delta = 0.95, ## Target acceptance rate
                   max_treedepth = 10) ## Maximum per-side steps (before U-turn)
  )
  saveRDS(model_fit, file = paste0("models/model_fit_prior_",prior_name,".RDS"))
  rm(model_fit, prior_name); closeAllConnections(); gc()
}

pars <- c("lambda", "phi", "tau", "delta", "alpha_r", "beta_r", "c_r",
          "alpha_s", "beta_s", "c_s", "c_c", "w", "pi_base", "pi_pert")

## Load models to get estimates
model_narrow <- readRDS(file = "models/model_fit_prior_narrow.RDS")
model_normal <- readRDS(file = "models/model_fit_prior_normal.RDS")
model_wide <- readRDS(file = "models/model_fit_prior_wide.RDS")

# -------------------------------------------------------------------------

posteriors <- list(
  narrow = extract(model_narrow, pars = pars) |> as.data.table(),
  normal = extract(model_normal, pars = pars) |> as.data.table(),
  wide = extract(model_wide, pars = pars) |> as.data.table()
) |> rbindlist(idcol = "prior")

pd_long <- melt.data.table(
  data = posteriors,
  id.vars = "prior"
)

levels(pd_long$variable) <- c("lambda","phi","tau","delta",
                              "alpha[r]", "beta[r]", "c[r]",
                              "alpha[s]", "beta[s]", "c[s]", "c[c]", "w",
                              "pi[base]~VLF", "pi[base]~LF", "pi[base]~HF",
                              "pi[pert]~VLF", "pi[pert]~LF", "pi[pert]~HF")
pd_long$prior <- factor(pd_long$prior,
       levels = c("narrow","normal","wide"),
       labels = c("Narrow", "Default", "Wide"))

fig <- ggplot(pd_long, aes(value, prior, fill = prior)) +
  facet_wrap(~ variable, scales = "free",
             labeller = label_parsed, ncol = 3) +
  ggdist::stat_halfeye(normalize = "groups", adjust = 3, show.legend = FALSE,
                       height = 0.5) +
  labs(x = "Parameter value", y = "Prior Width") +
  scale_fill_manual(values = c("#F67280", "#C06C84", "#355C7D")) +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(panel.grid.major.y = element_line(colour = "gray",
                                          linewidth = 1/3),
        strip.background = element_blank())

ggsave(filename = "figures/fig-prior-sensitivity.jpeg", fig,
       device = "jpeg", width = 9, height = 9, dpi = 500)
ggsave(filename = "figures/fig-prior-sensitivity.pdf", fig,
       device = "pdf", width = 9, height = 9)


# -------------------------------------------------------------------------

vs_narrow <- pd_long[, j = {
  diff <- value[prior == "Default"] - value[prior == "Narrow"]
  ci_num <- tidybayes::hdci(diff)
  estimate <- median(diff)
  scale <- sd(diff)

  list(Estimate = round(estimate, 2),
       `Lower bound` = round(ci_num[1], 2),
       `Upper bound` = round(ci_num[2], 2),
       pd = fifelse(estimate >= 0,
                                   mean(diff >= 0)*100,
                                   mean(diff <= 0)*100) |>
         round(2),
       ps = fifelse(estimate >= 0,
                                     mean(diff >= 0.1*scale)*100,
                                     mean(diff <= -0.1*scale)*100) |>
         round(2))
}, keyby = list(Parameter = variable)]

vs_wide <- pd_long[, j = {
  diff <- value[prior == "Default"] - value[prior == "Wide"]
  ci_num <- tidybayes::hdci(diff)
  estimate <- median(diff)
  scale <- sd(diff)

  list(Estimate = round(estimate, 2),
       `Lower bound` = round(ci_num[1], 2),
       `Upper bound` = round(ci_num[2], 2),
       pd = fifelse(estimate >= 0,
                    mean(diff >= 0)*100,
                    mean(diff <= 0)*100) |>
         round(2),
       ps = fifelse(estimate >= 0,
                    mean(diff >= 0.1*scale)*100,
                    mean(diff <= -0.1*scale)*100) |>
         round(2))
}, keyby = list(Parameter = variable)]

levels(vs_wide$Parameter) <-
  levels(vs_narrow$Parameter) <-
  c("$\\lambda$", "$\\phi$", "$\\tau$", "$\\delta$", "$\\alpha_r$",
  "$\\beta_r$", "$c_r$", "$\\alpha_s$", "$\\beta_s$", "$c_s$",
  "$c_c$", "$w$",
  "$\\vec{\\pi}_{base}$ [VLF]", "$\\vec{\\pi}_{base}$ [LF]", "$\\vec{\\pi}_{base}$ [HF]",
  "$\\vec{\\pi}_{pert}$ [VLF]", "$\\vec{\\pi}_{pert}$ [LF]", "$\\vec{\\pi}_{pert}$ [HF]")

knitr::kable(vs_narrow, escape = FALSE, align = "l") |>
  saveRDS(file = "tables/tbl-S1-narrow.RDS")

knitr::kable(vs_wide, escape = FALSE, align = "l") |>
  saveRDS(file = "tables/tbl-S1-wide.RDS")
