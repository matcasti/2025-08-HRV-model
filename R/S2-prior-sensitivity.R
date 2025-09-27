
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
model <- stan_model(file = "models/prior_custom_rri_model.stan.stan")

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
      "c_c", "w", "pi_base", "pi_pert"
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

## Load models to get estimates
model_narrow <- readRDS(file = "models/model_fit_prior_narrow.RDS")
model_normal <- readRDS(file = "models/model_fit_prior_normal.RDS")
model_wide <- readRDS(file = "models/model_fit_prior_wide.RDS")

# -------------------------------------------------------------------------

posteriors <- list(
  narrow = extract(model_narrow) |> as.data.table(),
  normal = extract(model_normal) |> as.data.table(),
  wide = extract(model_wide) |> as.data.table()
) |> rbindlist(idcol = "prior")

posteriors[, lp__ := NULL]

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

ggsave(filename = "figures/fig-prior-sensitivity.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-prior-sensitivity.pdf", fig,
       device = "pdf", width = 9, height = 9)

