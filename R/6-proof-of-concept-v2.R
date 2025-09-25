
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

if (!file.exists("models/model_fit_poc_2.RDS")) {

  # Prepare data ------------------------------------------------------------

  prior_params <- with(poc_data, estimate_RRi_curve(time, RRi))$parameters |> abs()

  N_sin <- 100
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
  model_fit <- rstan::vb(
    object = model,
    init = 0,
    output_samples = 5000,
    data = stan_data,
    iter = 200000,
    seed = 123,
    adapt_iter = 5000,
    tol_rel_obj = 0.001,
  )
  saveRDS(model_fit, file = "models/model_fit_poc_2.RDS")
} else {
  model_fit <- readRDS(file = "models/model_fit_poc_2.RDS")
}

# Reconstructed signal ----------------------------------------------------

w_param <- extract(model_fit, pars = "w")$w |> median()

mu_hat <- extract(model_fit, pars = "mu") |>
  as.data.table()

mu_hat <- transpose(mu_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

mu_hat[, time := gsub("mu.V", "", time) |> as.numeric()]
mu_hat[, draw := gsub("V", "", draw) |> as.numeric()]

mu_hat <- mu_hat[, list(mu = median(value)), keyby = time]

# SDNN extraction ---------------------------------------------------------

sd_hat <- extract(model_fit, pars = "SDNN_t") |>
  as.data.table()

sd_hat <- transpose(sd_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

sd_hat[, time := gsub("SDNN_t.V", "", time) |> as.numeric()]
sd_hat[, draw := gsub("V", "", draw) |> as.numeric()]

sd_hat <- sd_hat[, list(sdnn = median(value)), keyby = time]

# RR extraction ---------------------------------------------------------

rr_base_hat <- extract(model_fit, pars = "RR_baseline") |>
  as.data.table()

rr_base_hat <- transpose(rr_base_hat, keep.names = "time") |>
  melt.data.table(id.vars = "time",
                  variable.name = "draw")

rr_base_hat[, time := gsub("RR_baseline.V", "", time) |> as.numeric()]
rr_base_hat[, draw := gsub("V", "", draw) |> as.numeric()]

rr_base_hat <- rr_base_hat[, list(rr = median(value)), keyby = time]


# Spectral proportions ----------------------------------------------------

p_t_hat <- extract(model_fit, pars = "p_t")$p_t

p_t_bands <- vector("list", length = 3)
for(j in 1:3) {
  j_band <- p_t_hat[, , j] |>
    as.data.table() |>
    transpose(keep.names = "time") |>
    melt.data.table(id.vars = "time",
                    variable.name = "draw")

  j_band[, time := gsub("V", "", time) |> as.numeric()]
  j_band[, draw := gsub("V", "", draw) |> as.numeric()]

  p_t_bands[[j]] <- j_band[, list(estimate = median(value)), keyby = time]
}

p_t_bands <- p_t_bands |>
  rbindlist(idcol = "band")

p_t_bands[, band := factor(band, levels = 1:3, labels = c("VLF","LF","HF"))]

# Combining the datasets --------------------------------------------------

pred_data <- rr_base_hat[sd_hat, on = "time"][mu_hat, on = "time"]

# Generate plot -----------------------------------------------------------

fig_mu <- ggplot() +
  geom_line(aes(time, RRi, col = "Observed"), poc_data, linetype = 1) +
  geom_ribbon(aes(x = poc_data$time, fill = "Residual noise",
                  ymin = mu - ((1 - w_param) * sdnn),
                  ymax = mu + ((1 - w_param) * sdnn)),
              data = pred_data, alpha = 0.5) +
  geom_line(aes(x = poc_data$time, y = mu, col = "Estimated µ(t)"),
            data = pred_data) +
  scale_color_manual(values = c("Observed" = "gray",
                                "Estimated µ(t)" = "firebrick")) +
  scale_fill_manual(values = c("Residual noise" = "firebrick")) +
  labs(fill = "Shaded Area", col = "Line", subtitle = "Observed and reconstructed signal",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

fig_rr <- ggplot() +
  geom_ribbon(aes(x = poc_data$time, fill = "SDNN(t)",
                  ymin = rr - sdnn,
                  ymax = rr + sdnn),
              data = pred_data, alpha = 0.5) +
  geom_line(aes(x = poc_data$time, y = rr, col = "Baseline RR(t)"),
            data = pred_data, linewidth = 1) +
  scale_color_manual(values = c("Baseline RR(t)" = "firebrick")) +
  scale_fill_manual(values = c("SDNN(t)" = "firebrick")) +
  labs(fill = "Shaded Area", col = "Line", subtitle = "Baseline heart period",
       x = "Time (minutes)", y = "ms") +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

fig_spectral <- ggplot(p_t_bands, aes(time, estimate)) +
  geom_line(aes(color = band), linewidth = 1) +
  scale_color_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                     aesthetics = c("color")) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), n.breaks = 5, limits = 0:1) +
  labs(subtitle = "Spectral signature",
       x = "Time (minutes)", y = "Proportion of Power",
       color = "Color", fill = "Color") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

fig <- cowplot::plot_grid(fig_mu, fig_rr, fig_spectral,
                            labels = c("(A)","(B)","(C)"),
                            ncol = 1, nrow = 3, align = "hv", axis = "r")

fig

ggsave(filename = "figures/fig-poc-vb.svg", fig,
       device = "svg", width = 6, height = 9)
ggsave(filename = "figures/fig-poc-vb.pdf", fig,
       device = "pdf", width = 6, height = 9)
