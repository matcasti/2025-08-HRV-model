
# Prepare workspace -------------------------------------------------------

## Import libraries
library(data.table)
library(ggplot2)
library(rstan)

## Import models
models <- list(
  readRDS("models/model_fit_scenario_1.RDS") |>
    extract() |>
    as.data.table(),
  readRDS("models/model_fit_scenario_2.RDS") |>
    extract() |>
    as.data.table(),
  readRDS("models/model_fit_scenario_3.RDS") |>
    extract() |>
    as.data.table()
)w

## Import parameters used for simulation
simulation_parameters <- readRDS("~/Research/2025-08 HRV-model/data/simulation_parameters.RDS")
figures <- vector("list", length = 3)
for(i in 1:3) {
  sim <- models[[i]][, lambda:pi_pert.V3]
  sim[, row_id := seq_len(.N)]
  sim_long <- melt(sim, id.vars = "row_id")

  levels(sim_long$variable) <- c("lambda","phi","tau","delta",
                                "alpha[r]", "beta[r]", "c[r]",
                                "alpha[s]", "beta[s]", "c[s]", "c[c]", "w",
                                "pi[base]~VLF", "pi[base]~LF", "pi[base]~HF",
                                "pi[pert]~VLF", "pi[pert]~LF", "pi[pert]~HF")

  params <- unlist(simulation_parameters[[i]])[1:18] |>
    as.data.table(keep.rownames = TRUE) |>
    `names<-`(c("variable", "value"))

  params$variable <- levels(sim_long$variable)
  params$variable <- factor(params$variable, levels = params$variable)


  figures[[i]] <- ggplot(sim_long, aes(x = value, fill = variable)) +
    facet_wrap(~variable, ncol = 3, scales = "free_x", labeller = label_parsed) +
    ggdist::stat_halfeye(normalize = "panels", show.legend = FALSE,
                         adjust = 2) +
    scale_y_continuous(labels = NULL, breaks = NULL) +
    scale_x_continuous(n.breaks = 3) +
    geom_vline(data = params, aes(xintercept = value)) +
    scale_fill_viridis_d(option = "C", begin = 0.1, alpha = 0.9) +
    theme_classic(base_size = 12) +
    theme(strip.background = element_blank())
}

fig <- cowplot::plot_grid(plotlist = figures, align = "hv", ncol = 3,
                   labels = c("(A)","(B)","(C)"))

ggsave(filename = "figures/fig-full-recovery.svg", fig,
       device = "svg", width = 9, height = 9)
ggsave(filename = "figures/fig-full-recovery.pdf", fig,
       device = "pdf", width = 9, height = 9)
