
# Prepare workspace -------------------------------------------------------

## Load libraries
library(data.table)
library(rstan)
library(ggplot2)
library(CardioCurveR)

## Load model
model <- readRDS("models/model_fit_poc.RDS")
mu_hat <- as.data.table(model, pars = c("mu"))
w_hat <- as.data.table(model, pars = c("w"))$w |> median()
sdnn_hat <- as.data.table(model, pars = c("SDNN_t"))

## Load the data
poc_data <- CardioCurveR::import_RRi_txt(file = "data-raw/rri-jabf.txt",
                           remove_ectopic = TRUE,
                           filter_noise = FALSE) |>
  as.data.table()

rm(model); gc()

mu_hat[, row_id := seq_len(.N)]
sdnn_hat[, row_id := seq_len(.N)]

mu_hat <- melt(mu_hat, id.vars = "row_id", value.name = "mu")
sdnn_hat <- melt(sdnn_hat, id.vars = "row_id", value.name = "var")

mu_hat[, variable := gsub("mu\\[|\\]", "", variable) |> as.numeric()]
sdnn_hat[, variable := gsub("SDNN_t\\[|\\]", "", variable) |> as.numeric()]

ppcheck <- mu_hat[sdnn_hat, on = c("row_id", "variable")]

ppcheck[, RRi := rnorm(1, mu, sqrt((var^2) * (1 - w_hat))), by = list(row_id, variable)]

ppcheck[, time := poc_data$time[variable]]

ids <- sample.int(5000, 100)

fig <- ggplot() +
  geom_line(aes(time, RRi, group = row_id, col = "Predicted"), ppcheck[row_id %in% ids], alpha = 0.1) +
  geom_line(aes(time, RRi, col = "Observed"), poc_data, alpha = 0.8) +
  scale_color_manual(values = c("Predicted" = "dodgerblue", "Observed" = "gray20")) +
  labs(x = "Time (minutes)", y = "ms", color = "Line") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

ggsave(filename = "figures/fig-poc-ppcheck.jpeg", fig,
       device = "jpeg", width = 9, height = 9, dpi = 500)
ggsave(filename = "figures/fig-poc-ppcheck.pdf", fig,
       device = "pdf", width = 9, height = 9)


stats_model <- ppcheck[, list(mean = mean(RRi),
               sd = sd(RRi)), by = row_id]

stats_data <- poc_data[, list(mean = mean(RRi),
                sd = sd(RRi))]

fig <- ggplot() +
  geom_density2d_filled(aes(mean, sd), stats_model, show.legend = FALSE) +
  geom_point(aes(mean, sd), color = "white", stats_data) +
  geom_vline(aes(xintercept = mean), color = "white", stats_data, linetype = 2) +
  geom_hline(aes(yintercept = sd), color = "white", stats_data, linetype = 2) +
  labs(x = "Mean", y = "Standard deviation", color = "Color") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

ggsave(filename = "figures/fig-poc-stat-computed.jpeg", fig,
       device = "jpeg", width = 9, height = 9, dpi = 500)
ggsave(filename = "figures/fig-poc-stat-computed.pdf", fig,
       device = "pdf", width = 9, height = 9)
