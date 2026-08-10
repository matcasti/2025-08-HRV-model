# ---
# Continuous Wavelet Transform (CWT) HRV Analysis and Performance Evaluation
#
# Description:
# This script implements the CWT-based comparator referenced in Supplementary
# Section S4.2 and Table 3 of the main text: a second, independent
# conventional method (alongside the STFT/windowed approach in
# 2-classic_metrics.R) for estimating time-varying VLF/LF/HF power
# proportions from the same synthetic R-R interval data. It benchmarks the
# CWT reconstruction against q_j(t) (implied variance/power share), exactly
# as the STFT comparator is benchmarked, using the shared run_cwt_comparison()
# / block_bootstrap_metrics() helpers in R/_functions.R.
#
# Method: a Morlet continuous wavelet transform (Torrence & Compo 1998),
# implemented directly (see R/_functions.R::morlet_cwt for the validated
# implementation and normalization notes). Band power is integrated across
# each canonical band's frequency range at every (continuously resampled)
# time point and normalized to a proportion, using the same per-time-step
# normalization convention as the STFT comparator. A moving-average highpass
# filter (60s window, matching the primary STFT window) is applied before
# the transform; see R/_functions.R::get_cwt_band_proportions for why a
# single global linear detrend (adequate for STFT, which detrends
# independently within each short window) is NOT adequate here.
#
# Author: Matías Castillo-Aguilar
# ---

library(data.table)
library(ggplot2)
library(cowplot)
source("R/_functions.R")

simulated_data <- readRDS("data/simulated_data.RDS")

metrics <- vector("list", 3)
plots <- vector("list", 3)

N_FREQ <- 150       # scan frequencies across the full 0.003-0.4 Hz range
FS_CWT <- 4         # resampling rate (Hz) for the wavelet transform
N_BOOT <- 2000
BOOT_SEED <- 2025

for (i in 1:3) {

  sim_data <- simulated_data[[i]]$data

  # --- Run the CWT-based analysis ---
  comparison <- run_cwt_comparison(sim_data_i = sim_data, fs = FS_CWT, n_freq = N_FREQ)
  full_comparison_data <- comparison$estimates
  metrics[[i]]$estimates <- full_comparison_data
  metrics[[i]]$statistics <- comparison$statistics

  # --- Block-bootstrap 95% CIs (Table 3 uncertainty) ---
  metrics[[i]]$bootstrap <- list(
    vlf_metrics = block_bootstrap_metrics(full_comparison_data$q_vlf, full_comparison_data$p_vlf_cwt_interp, n_boot = N_BOOT, seed = BOOT_SEED),
    lf_metrics = block_bootstrap_metrics(full_comparison_data$q_lf, full_comparison_data$p_lf_cwt_interp, n_boot = N_BOOT, seed = BOOT_SEED),
    hf_metrics = block_bootstrap_metrics(full_comparison_data$q_hf, full_comparison_data$p_hf_cwt_interp, n_boot = N_BOOT, seed = BOOT_SEED)
  )

  # --- Visualize: spectral proportion reconstruction only (RR/SDNN are not
  # re-estimated by CWT; those comparisons are already covered by the STFT
  # comparator in Figure 4 / fig-windowed-method) ---
  legend <- FALSE
  if (i == 3) legend <- NA

  p_spectral <-
    full_comparison_data[, list(t, q_vlf, q_lf, q_hf,
                                p_vlf_cwt_interp, p_lf_cwt_interp,
                                p_hf_cwt_interp)] |>
    melt(id.vars = "t") |>
    (\(x){
      x[, Line := fifelse(grepl("interp", variable), "CWT estimate", "Ground truth")]
      x[, Band := fcase(
        grepl("^q_vlf|^p_vlf", variable), "VLF",
        grepl("^q_lf|^p_lf", variable), "LF",
        grepl("^q_hf|^p_hf", variable), "HF"
      )][]
    })() |>
    ggplot(aes(x = t, y = value, linetype = Line, color = Band)) +
    facet_grid(rows = vars(Band)) +
    geom_line(show.legend = legend) +
    scale_color_manual(values = c("HF" = "#0D1164", "LF" = "#640D5F", "VLF" = "#EA2264"),
                       aesthetics = c("color", "fill")) +
    scale_linetype_manual(values = c(6,1)) +
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(limits = 0:1, n.breaks = 5) +
    labs(subtitle = ifelse(i == 1, "Spectral signature (CWT)", ""),
         x = "Time (minutes)", y = "Proportion of Power", color = "Color", linetype = "Line") +
    theme_classic(base_size = 12)

  plots[[i]] <- p_spectral
}

# Point-estimate table
error_cwt <- lapply(1:3, function(i) {
  metrics[[i]]$statistics |> rbindlist(idcol = "Domain")
}) |> rbindlist(idcol = "Scenario")
names(error_cwt) <- c("Scenario", "Domain", "Metric", "Estimate")
saveRDS(error_cwt, file = "data/error_stats_cwt.RDS")

# Point estimate + 95% block-bootstrap CI table (feeds Table 3 assembly)
error_cwt_ci <- lapply(1:3, function(i) {
  metrics[[i]]$bootstrap |> rbindlist(idcol = "Domain")
}) |> rbindlist(idcol = "Scenario")
saveRDS(error_cwt_ci, file = "data/error_stats_cwt_ci.RDS")

fig <- ggpubr::ggarrange(plotlist = plots,
                         ncol = 3,
                         align = "hv",
                         labels = c("(A)", "(B)", "(C)"))

ggsave(filename = "figures/fig-cwt-method.jpeg", fig,
       device = "jpeg", width = 9, height = 4.5, dpi = 500)
ggsave(filename = "figures/fig-cwt-method.pdf", fig,
       device = "pdf", width = 9, height = 4.5)
