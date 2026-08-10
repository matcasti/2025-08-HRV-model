# ---
# STFT Window-Length Sensitivity Analysis
#
# Description:
# Supplementary Section S4.3.1: the primary windowed/STFT comparator in
# 2-classic_metrics.R uses a fixed 60-second window. This script repeats
# that analysis at 30s and 120s window lengths (holding all other settings
# fixed) to characterize how sensitive the windowed comparator's accuracy is
# to this choice, using the shared run_classic_comparison() helper so the
# alignment/metric logic is identical to the primary analysis.
#
# Author: Matías Castillo-Aguilar
# ---

library(data.table)
library(ggplot2)
source("R/_functions.R")

simulated_data <- readRDS("data/simulated_data.RDS")

WINDOW_LENGTHS <- c(30, 60, 120)
OVERLAP_PERC <- 0.99
SAMPLING_RATE_HZ <- 2

results <- vector("list", length(WINDOW_LENGTHS) * 3)
k <- 1

for (w in WINDOW_LENGTHS) {
  for (i in 1:3) {
    sim_data <- simulated_data[[i]]$data

    comparison <- run_classic_comparison(
      sim_data_i = sim_data,
      window_seconds = w,
      overlap_perc = OVERLAP_PERC,
      sampling_rate_hz = SAMPLING_RATE_HZ
    )

    stats_dt <- comparison$statistics |> rbindlist(idcol = "Domain")
    stats_dt[, `:=`(WindowSeconds = w, Scenario = i)]
    results[[k]] <- stats_dt
    k <- k + 1
  }
}

window_sensitivity <- rbindlist(results)
setnames(window_sensitivity, old = "Value", new = "Estimate")

saveRDS(window_sensitivity, file = "data/window_sensitivity.RDS")

# --- Summary table: RMSE by domain x window length, averaged across scenarios ---
window_sensitivity_summary <- window_sensitivity[
  Metric == "RMSE",
  list(Mean_RMSE = mean(Estimate), SD_RMSE = sd(Estimate)),
  keyby = list(Domain, WindowSeconds)
]
window_sensitivity_summary[, Domain := factor(Domain,
  levels = c("rr_metrics", "sdnn_metrics", "vlf_metrics", "lf_metrics", "hf_metrics"),
  labels = c("$RR(t_i)$", "$\\sigma_{total}(t_i)$", "VLF", "LF", "HF")
)]

window_sensitivity_summary <- window_sensitivity_summary[order(Domain, WindowSeconds)]

knitr::kable(window_sensitivity_summary,
             col.names = c("Domain", "Window (s)", "Mean RMSE (across scenarios)", "SD RMSE"),
             digits = 3, align = "l") |>
  saveRDS(file = "tables/tbl-S4.RDS")

# --- Figure: RMSE vs window length, faceted by domain ---
p_window <- ggplot(window_sensitivity_summary, aes(x = WindowSeconds, y = Mean_RMSE)) +
  geom_line(color = "dodgerblue") +
  geom_point(size = 2, color = "dodgerblue") +
  geom_errorbar(aes(ymin = pmax(0, Mean_RMSE - SD_RMSE), ymax = Mean_RMSE + SD_RMSE), width = 5) +
  facet_wrap(vars(Domain), scales = "free_y", nrow = 1) +
  scale_x_continuous(breaks = WINDOW_LENGTHS) +
  labs(x = "STFT window length (s)", y = "RMSE (mean \u00b1 SD across scenarios)") +
  theme_classic(base_size = 12)

ggsave(filename = "figures/fig-window-sensitivity.jpeg", p_window,
       device = "jpeg", width = 10, height = 3, dpi = 500)
ggsave(filename = "figures/fig-window-sensitivity.pdf", p_window,
       device = "pdf", width = 10, height = 3)
