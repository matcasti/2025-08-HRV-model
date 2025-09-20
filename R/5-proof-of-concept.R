
# -------------------------------------------------------------------------
# Prepare workspace -------------------------------------------------------
# -------------------------------------------------------------------------

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
points(poc_data, pch = 16, cex = 0.5)


# -------------------------------------------------------------------------
# Model fitting process ---------------------------------------------------
# -------------------------------------------------------------------------

if (!file.exists("models/model_fit_poc.RDS")) {

  # Prepare data ------------------------------------------------------------

  prior_params <- with(poc_data, estimate_RRi_curve(time, RRi))$parameters |> abs()

  N_sin <- 30
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

  # Fit the model -----------------------------------------------------------

  model <- stan_model(file = "models/rri_model.stan")
  model_fit <- rstan::sampling(
    object = model,
    init = 0,
    pars = c(
      "lambda","phi","tau","delta",
      "alpha_r","beta_r","c_r",
      "alpha_s","beta_s","c_s",
      "c_c", "w", "pi_base", "pi_pert",
      "mu","RR_baseline","SDNN_t"
    ),
    data = stan_data,
    iter = 10000, warmup = 5000,
    chains = 4, cores = 4,
    seed = 12345,
    control = list(adapt_delta = 0.95, ## Target acceptance rate
                   max_treedepth = 10) ## Maximum per-side steps (before U-turn)
  )
  saveRDS(model_fit, file = "models/model_fit_poc.RDS")
} else {
  model_fit <- readRDS(file = "models/model_fit_poc.RDS")
}

# Extract indices ---------------------------------------------------------

posterior <- extract(model_fit, pars = c(
  "lambda","phi","tau","delta",
  "alpha_r","beta_r","c_r",
  "alpha_s","beta_s","c_s",
  "c_c", "w", "pi_base", "pi_pert"
)) |> as.data.table()

posterior[, row_id := seq_len(length.out = .N)]

## Posterior predictive checks with 1000 random draws
spectral_data <- posterior[j = generate_rri_simulation(N = 1800,
  t_max = max(poc_data$time),
  N_sin = N_sin,
  seed = row_id,
  params = list(
    lambda = lambda, phi = phi, tau = tau, delta = delta,
    alpha_r = alpha_r, beta_r = beta_r, c_r = c_r,
    alpha_s = alpha_s, beta_s = beta_s, c_s = c_s,
    w = w, c_c = c_c,
    pi_base = c(pi_base.V1, pi_base.V2, pi_base.V3),
    pi_pert = c(pi_pert.V1, pi_pert.V2, pi_pert.V3),
    alpha_gp = c(1,1,1),
    rho_gp = c(1,1,1))
  )$data, keyby = row_id
  ][j = list(
      p_vlf_mu = median(p_vlf),
      p_vlf_hdi = diff(x = ggdist::hdci(p_vlf)[1,]),
      p_lf_mu = median(p_lf),
      p_lf_hdi = diff(x = ggdist::hdci(p_lf)[1,]),
      p_hf_mu = median(p_hf),
      p_hf_hdi = diff(x = ggdist::hdci(p_hf)[1,])
    ),
    keyby = list(t)]

# Reconstructed signal ----------------------------------------------------

mu_hat <- extract(model_fit, pars = "mu") |>
  as.data.table()

mu_hat <- transpose(mu_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

mu_hat[, time := gsub("mu.V", "", time) |> as.numeric()]
mu_hat[, draw := gsub("V", "", draw) |> as.numeric()]

mu_hat <- mu_hat[, list(mu = median(value), hdi = diff(ggdist::hdi(value)[1,])), keyby = time]


# SDNN extraction ---------------------------------------------------------

sd_hat <- extract(model_fit, pars = "SDNN_t") |>
  as.data.table()

sd_hat <- transpose(sd_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

sd_hat[, time := gsub("SDNN_t.V", "", time) |> as.numeric()]
sd_hat[, draw := gsub("V", "", draw) |> as.numeric()]

sd_hat <- sd_hat[, list(mu = median(value), hdi = diff(ggdist::hdi(value)[1,])), keyby = time]

# RR extraction ---------------------------------------------------------

rr_base_hat <- extract(model_fit, pars = "RR_baseline") |>
  as.data.table()

rr_base_hat <- transpose(rr_base_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

rr_base_hat[, time := gsub("RR_baseline.V", "", time) |> as.numeric()]
rr_base_hat[, draw := gsub("V", "", draw) |> as.numeric()]

rr_base_hat <- rr_base_hat[, list(mu = median(value), hdi = diff(ggdist::hdi(value)[1,])), keyby = time]


# Generate plot -----------------------------------------------------------

fig_obs <- ggplot(poc_data, aes(time, RRi)) +
  geom_line(linetype = 1) +
  labs(subtitle = "Observed RRi data",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12)

fig_mu <- ggplot() +
  geom_line(aes(time, RRi, col = "Observed"), poc_data, linetype = 1) +
  geom_ribbon(aes(x = poc_data$time, fill = "Estimated µ(t)",
                  ymin = mu - hdi,
                  ymax = mu + hdi),
              data = mu_hat, alpha = 0.5) +
  geom_line(aes(x = poc_data$time, y = mu, col = "Estimated µ(t)"),
            data = mu_hat, linewidth = 1) +
  scale_color_manual(values = c("Observed" = "gray",
                                "Estimated µ(t)" = "firebrick"),
                     aesthetics = c("fill", "color")) +
  labs(fill = "Line", col = "Line", subtitle = "Observed and reconstructed signal",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

fig_rr <- ggplot() +
  geom_ribbon(aes(x = poc_data$time, fill = "Baseline RR(t)",
                  ymin = mu - hdi,
                  ymax = mu + hdi),
              data = rr_base_hat, alpha = 0.5) +
  geom_line(aes(x = poc_data$time, y = mu, col = "Baseline RR(t)"),
            data = rr_base_hat, linewidth = 1) +
  scale_color_manual(values = c("Baseline RR(t)" = "dodgerblue"),
                     aesthetics = c("fill", "color")) +
  labs(fill = "Line", col = "Line", subtitle = "Baseline heart period",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

fig_sdnn <- ggplot() +
  geom_ribbon(aes(x = poc_data$time, fill = "SDNN(t)",
                  ymin = mu - hdi,
                  ymax = mu + hdi),
              data = sd_hat, alpha = 0.5) +
  geom_line(aes(x = poc_data$time, y = mu, col = "SDNN(t)"),
            data = sd_hat, linewidth = 1) +
  scale_color_manual(values = c("SDNN(t)" = "darkorange"),
                     aesthetics = c("fill", "color")) +
  labs(fill = "Line", col = "Line", subtitle = "Instantaneous signal variability",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

predicted_spectral <- melt(
  data = spectral_data,
  id.vars = "t",
  measure.vars = list(mu = c("p_vlf_mu", "p_lf_mu", "p_hf_mu"),
                      hdi = c("p_vlf_hdi", "p_lf_hdi", "p_hf_hdi"))
)
predicted_spectral[, variable := factor(variable,
                                        levels = 1:3,
                                        labels = c("VLF", "LF", "HF"))]

fig_spectral <- ggplot() +
  geom_ribbon(mapping = aes(x = t, fill = variable,
                            ymin = mu - hdi,
                            ymax = mu + hdi),
              data = predicted_spectral, alpha = 0.3) +
  geom_line(mapping = aes(t, mu, color = variable),
            data = predicted_spectral, linewidth = 1) +
  scale_color_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                     aesthetics = c("color", "fill")) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), n.breaks = 5) +
  labs(subtitle = "Spectral signature",
       x = "Time (minutes)", y = "Proportion of Power",
       color = "Color", fill = "Color") +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom")

plots <- cowplot::plot_grid(fig_rr, fig_sdnn, fig_spectral,
                            labels = c("(B)","(C)","(D)"),
                            ncol = 3, nrow = 1, align = "hv", axis = "l")

fig <- cowplot::plot_grid(fig_mu, plots, nrow = 2, rel_heights = c(0.5, 0.5),
                          labels = c("(A)",""))

ggsave(filename = "figures/fig-poc.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-poc.pdf", fig,
       device = "pdf", width = 9, height = 9)

names(posterior)[5:18] <- c("alpha[r]", "beta[r]", "c[r]", "alpha[s]", "beta[s]", "c[s]", "c[c]",
                            "w", "pi[base]~VLF", "pi[base]~LF", "pi[base]~HF", "pi[pert]~VLF",
                            "pi[pert]~LF", "pi[pert]~HF")

posterior[, chain := rep(1:4, each = 5000)]
posterior[, row := 1:5000, by = chain]

fig <- melt.data.table(posterior, id.vars = c("chain", "row_id", "row")) |>
  ggplot(aes(value)) +
  facet_wrap(~variable, ncol = 3, scales = "free",
             labeller = label_parsed, strip.position = "top") +
  ggdist::stat_halfeye(aes(fill = variable), normalize = "panels", show.legend = FALSE) +
  scale_x_continuous(expand = c(0.05,0), breaks = scales::breaks_extended(n = 5)) +
  scale_y_continuous(expand = c(0.15,0,0,0), breaks = NULL, name = NULL) +
  scale_fill_viridis_d(option = "B",begin = 0.1, end = 0.9, alpha = 0.8) +
  labs(x = "Parameter value", color = "Chain",
       subtitle = "Posterior distribution of model parameters") +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        panel.spacing.x = unit(5, "mm"))

ggsave(filename = "figures/fig-poc-posterior.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-poc-posterior.pdf", fig,
       device = "pdf", width = 9, height = 9)
