
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(rstan)
library(ggplot2)

## Load auxiliary functions
source("R/_functions.R")

## Load the model
model <- stan_model(file = "models/ppc_rri_model.stan")

## Prepare the simulation data
stan_data <- list(
  N = 1200,
  t = seq(0, 15, length.out = 1200),
  N_sin = 30,
  freqs = list(
    seq(0.003, 0.039, length.out = 30), # VLF
    seq(0.040, 0.149, length.out = 30), # LF
    seq(0.150, 0.400, length.out = 30)  # HF
  ),
  tau_mu = 6,
  delta_mu = 3,
  lambda_mu = 5,
  phi_mu = 3,
  rr_min_hypothetical = 350,
  rr_range_hypothetical = 450,
  rr_sd_hypothetical = 200
)

## Compute prior predictive samples
prior_fit <- sampling(
  object = model,
  pars = c(
    "lambda","phi","tau","delta",
    "alpha_r","beta_r","c_r",
    "alpha_s","beta_s","c_s",
    "c_c", "w", "pi_base", "pi_pert",
    "RR_prior","mu","RR_baseline","SDNN_t"
  ),
  data = stan_data,
  iter = 10000, warmup = 5000,
  chains = 4, cores = 4,
  seed = 12345
)


# Inspecting the samples --------------------------------------------------

ppc_rri <- extract(prior_fit, pars = "RR_prior") |>
  as.data.table() |>
  transpose(keep.names = "Time") |>
  melt.data.table(id.vars = "Time")

ppc_rri[, Time := gsub("*.+\\.V", "", Time) |> as.numeric()]
ppc_rri[, variable := gsub("V", "", variable) |> as.numeric()]

ppc_rri_estimate <- ppc_rri[, {
  mu = median(value)
  hdi = ggdist::hdci(value)
  list(mu = mu, ci_low = hdi[1,1], ci_high = hdi[1,2])
}, keyby = Time]

fig_rri <- ggplot(ppc_rri_estimate, aes(x = stan_data$t)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.3, fill = "dodgerblue") +
  geom_line(aes(y = mu), linewidth = 1, col = "dodgerblue") +
  scale_x_continuous(expand = c(0,0)) +
  labs(subtitle = "Prior RRi Signal",
       x = "Time (minutes)", y = "ms") +
  theme_classic(base_size = 12)

# -------------------------------------------------------------------------

ppc_mu <- extract(prior_fit, pars = "mu") |>
  as.data.table() |>
  transpose(keep.names = "Time") |>
  melt.data.table(id.vars = "Time")

ppc_mu[, Time := gsub("*.+\\.V", "", Time) |> as.numeric()]
ppc_mu[, variable := gsub("V", "", variable) |> as.numeric()]

ppc_mu_estimate <- ppc_mu[, {
  mu = median(value)
  hdi = ggdist::hdci(value)
  list(mu = mu, ci_low = hdi[1,1], ci_high = hdi[1,2])
}, keyby = Time]

fig_mu <- ggplot(ppc_mu_estimate, aes(x = stan_data$t)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.3, fill = "violet") +
  geom_line(aes(y = mu), linewidth = 1, col = "violet") +
  scale_x_continuous(expand = c(0,0)) +
  labs(subtitle = "Prior µ(t)",
       x = "Time (minutes)", y = "ms") +
  theme_classic(base_size = 12)

# -------------------------------------------------------------------------

ppc_rrbase <- extract(prior_fit, pars = "RR_baseline") |>
  as.data.table() |>
  transpose(keep.names = "Time") |>
  melt.data.table(id.vars = "Time")

ppc_rrbase[, Time := gsub("*.+\\.V", "", Time) |> as.numeric()]
ppc_rrbase[, variable := gsub("V", "", variable) |> as.numeric()]

ppc_rrbase_estimate <- ppc_rrbase[, {
  mu = median(value)
  hdi = ggdist::hdci(value)
  list(mu = mu, ci_low = hdi[1,1], ci_high = hdi[1,2])
}, keyby = Time]

fig_rrbase <- ggplot(ppc_rrbase_estimate, aes(x = stan_data$t)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.3, fill = "darkorange") +
  geom_line(aes(y = mu), linewidth = 1, col = "darkorange") +
  scale_x_continuous(expand = c(0,0)) +
  labs(subtitle = "Prior RR baseline",
       x = "Time (minutes)", y = "ms") +
  theme_classic(base_size = 12)

# -------------------------------------------------------------------------

ppc_sdnn <- extract(prior_fit, pars = "SDNN_t") |>
  as.data.table() |>
  transpose(keep.names = "Time") |>
  melt.data.table(id.vars = "Time")

ppc_sdnn[, Time := gsub("*.+\\.V", "", Time) |> as.numeric()]
ppc_sdnn[, variable := gsub("V", "", variable) |> as.numeric()]

ppc_sdnn_estimate <- ppc_sdnn[, {
  mu = median(value)
  hdi = ggdist::hdci(value)
  list(mu = mu, ci_low = hdi[1,1], ci_high = hdi[1,2])
}, keyby = Time]

fig_sdnn <- ggplot(ppc_sdnn_estimate, aes(x = stan_data$t)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.3, fill = "darkgreen") +
  geom_line(aes(y = mu), linewidth = 1, col = "darkgreen") +
  scale_x_continuous(expand = c(0,0)) +
  labs(subtitle = "Prior SDNN",
       x = "Time (minutes)", y = "ms") +
  theme_classic(base_size = 12)

# -------------------------------------------------------------------------

ppc_pj <- extract(prior_fit, pars = c(
  "lambda","phi","tau","delta",
  "alpha_r","beta_r","c_r",
  "alpha_s","beta_s","c_s",
  "c_c", "w", "pi_base", "pi_pert"
)) |> as.data.table()

ppc_pj[, row_id := seq_len(length.out = .N)]

ppc_pj_estimate <- ppc_pj[j = generate_rri_simulation(
  N = stan_data$N,
  t_max = 15,
  N_sin = 30,
  seed = row_id,
  params = list(
    lambda = lambda, phi = phi, tau = tau, delta = delta,
    alpha_r = alpha_r, beta_r = beta_r, c_r = c_r,
    alpha_s = alpha_s, beta_s = beta_s, c_s = c_s,
    w = w, c_c = c_c,
    pi_base = c(pi_base.V1, pi_base.V2, pi_base.V3),
    pi_pert = c(pi_pert.V1, pi_pert.V2, pi_pert.V3),
    rho_gp = c(1,1,1) * 0.5)
)$data, keyby = row_id
][j = {
  p_vlf_mu = median(p_vlf)
  p_vlf_hdi = ggdist::hdci(p_vlf)
  p_lf_mu = median(p_lf)
  p_lf_hdi = ggdist::hdci(p_lf)
  p_hf_mu = median(p_hf)
  p_hf_hdi = ggdist::hdci(p_hf)

  list(
    p_vlf_mu = p_vlf_mu, p_vlf_low = p_vlf_hdi[1,1], p_vlf_high = p_vlf_hdi[1,2],
    p_lf_mu = p_vlf_mu, p_lf_low = p_lf_hdi[1,1], p_lf_high = p_lf_hdi[1,2],
    p_hf_mu = p_vlf_mu, p_hf_low = p_hf_hdi[1,1], p_hf_high = p_hf_hdi[1,2]
  )
},
keyby = list(t)]

ppc_pj_estimate <- melt(
  data = ppc_pj_estimate,
  id.vars = "t",
  measure.vars = list(mu = c("p_vlf_mu", "p_lf_mu", "p_hf_mu"),
                      ci_low = c("p_vlf_low", "p_lf_low", "p_hf_low"),
                      ci_high = c("p_vlf_high", "p_lf_high", "p_hf_high"))
)
ppc_pj_estimate[, variable := factor(variable,
                                     levels = 1:3,
                                     labels = c("VLF", "LF", "HF"))]

fig_pj <- ggplot(ppc_pj_estimate) +
  facet_wrap(~variable, ncol = 1) +
  geom_ribbon(mapping = aes(x = t, fill = variable,
                            ymin = ci_low,
                            ymax = ci_high),
              alpha = 0.3, show.legend = FALSE) +
  geom_line(mapping = aes(t, mu, color = variable),
            linewidth = 1, show.legend = FALSE) +
  scale_color_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                     aesthetics = c("color", "fill")) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), n.breaks = 5, limits = 0:1) +
  labs(subtitle = "Prior spectral signature",
       x = "Time (minutes)", y = "Proportion of Power",
       color = "Color", fill = "Color") +
  theme_classic(base_size = 12) +
  theme(legend.position = "bottom",
        strip.background = element_blank())

fig_bottom <- cowplot::plot_grid(fig_rrbase, fig_sdnn, fig_pj,
                   labels = c("(B)","(C)","(D)"),
                   ncol = 3, nrow = 1, align = "hv", axis = "l")

fig <- cowplot::plot_grid(fig_rri, fig_bottom, nrow = 2, rel_heights = c(0.5, 0.5),
                          labels = c("(A)",""))

ggsave(filename = "figures/fig-ppcheck.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-ppcheck.pdf", fig,
       device = "pdf", width = 9, height = 9)
