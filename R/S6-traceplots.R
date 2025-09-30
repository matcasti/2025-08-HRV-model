
# Prepare workspace -------------------------------------------------------

## Load libraries
library(rstan)
library(data.table)
library(ggplot2)

## Load model
model <- readRDS("models/model_fit_prior_normal.RDS")

posterior <- as.data.table(model, pars = c(
  "lambda","phi","tau","delta",
  "alpha_r","beta_r","c_r",
  "alpha_s","beta_s","c_s",
  "c_c", "w", "pi_base", "pi_pert"
))

names(posterior) <- c("lambda","phi","tau","delta",
                      "alpha[r]", "beta[r]", "c[r]",
                      "alpha[s]", "beta[s]", "c[s]", "c[c]", "w",
                      "pi[base]~VLF", "pi[base]~LF", "pi[base]~HF",
                      "pi[pert]~VLF", "pi[pert]~LF", "pi[pert]~HF")

posterior[, row_id := seq_len(.N)]
posterior[, chain := rep(1:4, each = 5000)]
posterior[, row := 1:5000, by = chain]

posterior_long <- melt.data.table(posterior,
                id.vars = c("row_id", "chain", "row"))


fig <- ggplot(posterior_long, aes(y = value, x = row)) +
  facet_wrap(~ variable, scales = "free_y", ncol = 3, labeller = label_parsed) +
  geom_line(aes(col = as.ordered(chain))) +
  scale_color_viridis_d(option = "F") +
  labs(x = "Iterations", y = "Parameter value", col = "Chain") +
  theme_classic(base_size = 12) +
  theme(strip.background = element_blank())

ggsave(filename = "figures/fig-poc-traceplot.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-poc-traceplot.pdf", fig,
       device = "pdf", width = 9, height = 9)
