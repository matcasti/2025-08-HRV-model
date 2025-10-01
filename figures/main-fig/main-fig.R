

# Prepare workspace -------------------------------------------------------

## Load libraries
library(ggplot2)
library(cowplot)
library(data.table)

# -------------------------------------------------------------------------

sim_data <- readRDS("data/simulated_data.RDS")[[1]]

theme_set(new = theme_classic(base_size = 12))

# Time-domain curves ------------------------------------------------------

fig_rr <- ggplot(sim_data$data) +
  geom_line(aes(t, RR_baseline), linewidth = 1, col = "firebrick") +
  labs(x = "Time (min)", y = "ms") +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0.1,0)) +
  theme(axis.ticks = element_blank())

fig_amp <- ggplot(sim_data$data) +
  geom_line(aes(t, A_t), linewidth = 1, col = "darkorange") +
  geom_line(aes(t, -A_t), linewidth = 1, col = "darkorange") +
  geom_hline(aes(yintercept = 0), linetype = 2) +
  labs(x = "Time (min)", y = "ms") +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL) +
  theme(axis.ticks = element_blank())

time_domain_fig <- plot_grid(fig_rr, fig_amp, nrow = 2, align = "v", axis = "l")

# Frequency curves --------------------------------------------------------

fig_vlf_wave <- ggplot(sim_data$data) +
  geom_line(aes(t, sin(10 * pi * t * 0.04)), linewidth = 1, col = "#0D1164") +
  labs(x = "Time (min)", y = "ms") +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0.1,0)) +
  theme(axis.ticks = element_blank())

fig_lf_wave <- ggplot(sim_data$data) +
  geom_line(aes(t, sin(10 * pi * t * 0.1)), linewidth = 1, col = "#640D5F") +
  labs(x = "Time (min)", y = "ms") +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0.1,0)) +
  theme(axis.ticks = element_blank())

fig_hf_wave <- ggplot(sim_data$data) +
  geom_line(aes(t, sin(10 * pi * t * 0.2)), linewidth = 1, col = "#EA2264") +
  labs(x = "Time (min)", y = "ms") +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0.1,0)) +
  theme(axis.ticks = element_blank())

freq_wave_fig <- plot_grid(fig_vlf_wave, fig_lf_wave, fig_hf_wave, nrow = 3, align = "v", axis = "l")

set.seed(1234)
vlf_curve <- exp(cumsum(x = rnorm(100, mean = 0, sd = 0.1)))
lf_curve <- exp(cumsum(x = rnorm(100, mean = 0, sd = 0.1)))
hf_curve <- exp(cumsum(x = rnorm(100, mean = 0, sd = 0.1)))

curve_data <- data.frame(
  t = 1:100,
  vlf_curve,
  lf_curve,
  hf_curve
)

curve_data$vlf_curve <- predict(mgcv::gam(vlf_curve ~ s(t), data = curve_data))
curve_data$lf_curve <- predict(mgcv::gam(lf_curve ~ s(t), data = curve_data))
curve_data$hf_curve <- predict(mgcv::gam(hf_curve ~ s(t), data = curve_data))

fig_vlf_amp <- ggplot(curve_data) +
  geom_area(aes(t, vlf_curve), linewidth = 1, col = "#0D1164", fill = "#0D1164", alpha = 0.8) +
  labs(x = "Frequency", y = expression(ms^2)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0,0,0.5,0)) +
  theme(axis.ticks = element_blank())

fig_lf_amp <- ggplot(curve_data) +
  geom_area(aes(t, lf_curve), linewidth = 1, col = "#640D5F", fill = "#640D5F", alpha = 0.8) +
  labs(x = "Frequency", y = expression(ms^2)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0,0,0.5,0)) +
  theme(axis.ticks = element_blank())

fig_hf_amp <- ggplot(curve_data) +
  geom_area(aes(t, hf_curve), linewidth = 1, col = "#EA2264", fill = "#EA2264", alpha = 0.8) +
  labs(x = "Frequency", y = expression(ms^2)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  scale_y_continuous(labels = NULL, expand = c(0,0,0.5,0)) +
  theme(axis.ticks = element_blank())

freq_amp_fig <- plot_grid(fig_vlf_amp, fig_lf_amp, fig_hf_amp, nrow = 3, align = "v", axis = "l")

# Figure band proportions -------------------------------------------------

long_spec <- melt(sim_data$data, id.vars = "t", measure.vars = c("p_vlf", "p_lf", "p_hf"))
long_spec[, variable := `levels<-`(variable, c("VLF", "LF", "HF"))]

fig_band_prop <- ggplot(long_spec, aes(t, value)) +
  geom_area(aes(fill = variable, color = variable), alpha = 0.8, linewidth = 1, show.legend = FALSE) +
  scale_color_manual(values = c("#0D1164", "#640D5F", "#EA2264")) +
  scale_fill_manual(values = c("#0D1164", "#640D5F", "#EA2264")) +
  scale_y_continuous(expand = c(0,0), labels = NULL) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  labs(color = "Band", fill = "Band", x = "Time (min)", y = "%") +
  theme(axis.ticks = element_blank())


fig_band_vlf <- ggplot(sim_data$data, aes(t, p_vlf)) +
  geom_area(fill = "#0D1164", col = "#0D1164", linewidth = 1, alpha = 0.8) +
  scale_y_continuous(labels = NULL, expand = c(0,0), limits = c(0,1)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  labs(x = "Time (min)", y = "%") +
  theme(axis.ticks = element_blank())

fig_band_lf <- ggplot(sim_data$data, aes(t, p_lf)) +
  geom_area(fill = "#640D5F", col = "#640D5F", linewidth = 1, alpha = 0.8) +
  scale_y_continuous(labels = NULL, expand = c(0,0), limits = c(0,1)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  labs(x = "Time (min)", y = "%") +
  theme(axis.ticks = element_blank())

fig_band_hf <- ggplot(sim_data$data, aes(t, p_hf)) +
  geom_area(fill = "#EA2264", col = "#EA2264", linewidth = 1, alpha = 0.8) +
  scale_y_continuous(labels = NULL, expand = c(0,0), limits = c(0,1)) +
  scale_x_continuous(labels = NULL, expand = c(0,0)) +
  labs(x = "Time (min)", y = "%") +
  theme(axis.ticks = element_blank())

freq_prop_fig <- plot_grid(
  fig_band_prop,
  plot_grid(fig_band_vlf, fig_band_lf, fig_band_hf, nrow = 3, align = "v", axis = "l"),
  ncol = 2
)


# Residual error ----------------------------------------------------------

fig_residual <- ggplot() +
  geom_line(aes(1:500, rnorm(500)), col = "gray") +
  geom_hline(yintercept = 0, linetype = 2) +
  scale_x_continuous(expand = c(0,0), labels = NULL) +
  scale_y_continuous(expand = c(1,0), labels = NULL) +
  labs(y = "ms", x = "Time (min)") +
  theme(axis.ticks = element_blank())


# Assemble images ---------------------------------------------------------

fig_main <- plot_grid(
  time_domain_fig,
  plot_grid(freq_wave_fig, freq_amp_fig, ncol = 2),
  freq_prop_fig,
  plot_grid(fig_residual),
  nrow = 4, rel_heights = c(3/4, 3/4, 3/4, 2/4), align = "hv", axis = "lrtb"
)

ggsave(filename = "figures/main-fig/fig_main_solo.svg", plot = fig_main,
       width = 6, height = 12)
