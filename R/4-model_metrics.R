

# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(rstan)
library(ggplot2)

## Load functions
source("R/_functions.R")

## Load simulated data
sim_data <- readRDS("data/simulated_data.RDS")
params <- readRDS("data/simulation_parameters.RDS")

## Load models
models <- lapply(1:3, function(i) {
  file_path <- paste0("models/model_fit_scenario_",i,".RDS")
  readRDS(file_path)
})

tables <- vector("list", length = 3)
for (i in 1:3) {
  tbl <- bayestestR::describe_posterior(
    posterior = models[[i]],
    ci_method = "HDI",
    test = NULL,
    diagnostic = NULL) |>
    as.data.table()

  tables[[i]] <- tbl[
    i = !Parameter %like% "_log|_logit",
    list(Parameter = if(i == 1) Parameter else NULL,
         Truth = NA,
         Estimate = round(Median, 2),
         `95% CI` = paste0("[", round(CI_low, 2), ", ", round(CI_high, 2), "]"))
  ]
}

do.call(cbind, tables)

# Extract model posterior distribution ------------------------------------

posteriors <- lapply(models, function(x) {
  X <- as.data.table(x = extract(x))
  X[, row_id := seq_len(.N)][]
})

# Compute predicted RRi curve ---------------------------------------------

if (!file.exists("data/model_predictions.RDS")) {
  predicted <- lapply(posteriors, function(x) {
    x[j = generate_rri_simulation(
      N = 1800,
      t_max = 15,
      N_sin = 25,
      seed = 123,
      params = list(
        lambda = lambda, phi = phi, tau = tau, delta = delta,
        alpha_r = alpha_r, beta_r = beta_r, c_r = c_r,
        alpha_s = alpha_s, beta_s = beta_s, c_s = c_s,
        w = w, c_c = c_c,
        pi_base = c(pi_base.V1, pi_base.V2, pi_base.V3),
        pi_pert = c(pi_pert.V1, pi_pert.V2, pi_pert.V3),
        alpha_gp = c(1,1,1), rho_gp = c(1,1,1)
      )
    )$data,
    keyby = row_id
    ][
      j = list(
        RR_mu = median(RR),
        RR_hdi = diff(x = ggdist::hdci(RR)[1,]),
        mu_mu = median(mu),
        mu_hdi = diff(x = ggdist::hdci(mu)[1,]),
        RR_baseline_mu = median(RR_baseline),
        RR_baseline_hdi = diff(x = ggdist::hdci(RR_baseline)[1,]),
        SDNN_t_mu = median(SDNN_t),
        SDNN_t_hdi = diff(x = ggdist::hdci(SDNN_t)[1,]),
        p_vlf_mu = median(p_vlf),
        p_vlf_hdi = diff(x = ggdist::hdci(p_vlf)[1,]),
        p_lf_mu = median(p_lf),
        p_lf_hdi = diff(x = ggdist::hdci(p_lf)[1,]),
        p_hf_mu = median(p_hf),
        p_hf_hdi = diff(x = ggdist::hdci(p_hf)[1,])
      ),
      keyby = list(t)
    ]
  })

  saveRDS(predicted, file = "data/model_predictions.RDS")
} else {
  predicted <- readRDS("data/model_predictions.RDS")
}

plots <- vector("list", length = 3)
for (i in 1) {
  legend <- FALSE
  if (i == 3) {
    legend <- NA
  }

  fig_rr <- ggplot() +
    geom_ribbon(mapping = aes(x = t, fill = "Model estimate",
                              ymin = RR_baseline_mu - RR_baseline_hdi,
                              ymax = RR_baseline_mu + RR_baseline_hdi),
                data = predicted[[i]], show.legend = legend, alpha = 0.5) +
    geom_line(mapping = aes(t, RR_baseline_mu, color = "Model estimate"),
              data = predicted[[i]], show.legend = legend) +
    geom_line(mapping = aes(t, RR_baseline, color = "Ground truth"),
              data = sim_data[[i]]$data, linetype = 6, show.legend = legend) +
    scale_color_manual(values = c("Ground truth" = "black",
                                  "Model estimate" = "dodgerblue"),
                       aesthetics = c("fill", "color")) +
    scale_y_continuous(n.breaks = 4) +
    labs(subtitle = ifelse(i == 1, "Signal trajectory", ""), color = "Line",
         x = "Time (minutes)", y = "ms", fill = "Line") +
    theme_classic(base_size = 12)

  fig_sdnn <- ggplot() +
    geom_ribbon(mapping = aes(x = t, fill = "Model estimate",
                              ymin = SDNN_t_mu - SDNN_t_hdi,
                              ymax = SDNN_t_mu + SDNN_t_hdi),
                data = predicted[[i]], show.legend = legend, alpha = 0.5) +
    geom_line(mapping = aes(t, SDNN_t_mu, color = "Model estimate"),
              data = predicted[[i]], show.legend = legend) +
    geom_line(mapping = aes(t, SDNN_t, color = "Ground truth"),
              data = sim_data[[i]]$data, linetype = 5, show.legend = legend) +
    scale_color_manual(values = c("Ground truth" = "black",
                                  "Model estimate" = "darkorange"),
                       aesthetics = c("fill", "color")) +
    scale_y_continuous(n.breaks = 5) +
    labs(subtitle = ifelse(i == 1, "Signal SDNN", ""), color = "Line",
         x = "Time (minutes)", y = "ms", fill = "Line") +
    theme_classic(base_size = 12)

  sim_data_spectral <- melt(
    data = sim_data[[i]]$data,
    id.vars = "t",
    measure.vars = c("p_vlf", "p_lf", "p_hf")
  )
  sim_data_spectral[, variable := factor(variable,
                                         levels = c("p_vlf", "p_lf", "p_hf"),
                                         labels = c("VLF", "LF", "HF"))]
  predicted_spectral <- melt(
    data = predicted[[i]],
    id.vars = "t",
    measure.vars = list(mu = c("p_vlf_mu", "p_lf_mu", "p_hf_mu"),
                        hdi = c("p_vlf_hdi", "p_lf_hdi", "p_hf_hdi"))
  )
  predicted_spectral[, variable := factor(variable,
                                          levels = 1:3,
                                          labels = c("VLF", "LF", "HF"))]

  fig_spectral <- ggplot() +
    facet_grid(rows = vars(variable)) +
    geom_ribbon(mapping = aes(x = t, fill = variable,
                              ymin = mu - hdi,
                              ymax = mu + hdi),
                data = predicted_spectral, show.legend = legend, alpha = 0.5) +
    geom_line(mapping = aes(t, mu, color = variable, linetype = "Model estimate"),
              data = predicted_spectral, show.legend = legend) +
    geom_line(mapping = aes(t, value, color = variable, linetype = "Ground truth"),
              data = sim_data_spectral, show.legend = legend) +
    scale_color_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                       aesthetics = c("color", "fill")) +
    scale_linetype_manual(values = c("Ground truth" = 6, "Model estimate" = 1)) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(n.breaks = 5) +
    labs(subtitle = ifelse(i == 1, "Spectral signature", ""),
         x = "Time (minutes)", y = "Proportion of Power",
         color = "Color", linetype = "Line", fill = "Color") +
    theme_classic(base_size = 12)

  plots[[i]] <- cowplot::plot_grid(fig_rr, fig_sdnn, fig_spectral, ncol = 1, rel_heights = c(0.6,0.6,1), align = "hv", axis = "l")
}

fig <- ggpubr::ggarrange(plotlist = plots,
                         ncol = 3,
                         align = "hv",
                         widths = c(2,2,3.3),
                         labels = c("(A)", "(B)", "(C)"))

ggsave(filename = "figures/fig-model-method.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-model-method.pdf", fig,
       device = "pdf", width = 9, height = 9)
