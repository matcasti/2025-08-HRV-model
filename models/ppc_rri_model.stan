// Stan Model Adapted for Prior Predictive Sampling
//
// This version of the model is designed to sample ONLY from the prior distributions.
// It does not fit to any observed data. Its purpose is to generate synthetic
// datasets to visualize the implications of the chosen priors and model structure.
//
// Key modifications:
// 1. The observed data vector `RR` has been removed from the `data` block.
// 2. The likelihood statement `RR ~ normal(...)` has been removed from the `model` block.
// 3. All generative logic has been moved to the `generated quantities` block.
// 4. A new variable `RR_prior` is generated to hold the synthetic time series.

// =====================================================================
// Functions Block (Unchanged)
// =====================================================================
functions {
  vector logistic_curve(vector t, real location, real rate) {
    return inv_logit(rate * (t - location));
  }
}

// =====================================================================
// Data Block (Modified)
// =====================================================================
// Note: `RR` is no longer an input. The model only needs structural information.
data {
  // --- Structural Data ---
  int<lower=1> N;              // Number of data points in the time series.
  vector[N] t;                 // Time vector (e.g., in minutes).

  // --- Fixed Spectral Components (treated as data) ---
  int<lower=1> N_sin;          // Number of sinusoids per frequency band.
  array[3] vector[N_sin] freqs;  // Pre-calculated frequencies for VLF, LF, and HF bands.

  // --- Informative Priors on Double-Logistic (DL) parameters ---
  real<lower=0> tau_mu;        // Expected time of the perturbation event.
  real<lower=0> delta_mu;      // Expected delay between perturbation and recovery.
  real<lower=0> lambda_mu;     // Expected rate of the perturbation.
  real<lower=0> phi_mu;        // Expected rate of the recovery.

  // --- Data for Scaling (needed for parameter transformations) ---
  // In a real prior predictive check, you might pass hypothetical min/max/sd values.
  real rr_min_hypothetical;
  real rr_range_hypothetical;
  real rr_sd_hypothetical;
}

// =====================================================================
// Transformed Data Block (Unchanged)
// =====================================================================
transformed data {
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // --- Precompute sine and cosine basis function templates ---
  array[3] matrix[N, N_sin] sin_mat;
  array[3] matrix[N, N_sin] cos_mat;
  for (j in 1:3) {
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]';
    sin_mat[j] = sin(2 * pi() * T_mat);
    cos_mat[j] = cos(2 * pi() * T_mat);
  }

  // --- Precompute log-frequencies ---
  array[3] vector[N_sin] log_freqs;
  for (j in 1:3) log_freqs[j] = log(freqs[j]);

  // --- Precompute per-band Gram matrices ---
  array[3] matrix[N_sin, N_sin] G_sin;
  array[3] matrix[N_sin, N_sin] G_cos;
  array[3] matrix[N_sin, N_sin] G_sin_cos;
  real normalization = 1.0 / (N - 1);
  for (j in 1:3) {
    G_sin[j]       = (transpose(sin_mat[j]) * sin_mat[j]) * normalization;
    G_cos[j]       = (transpose(cos_mat[j]) * cos_mat[j]) * normalization;
    G_sin_cos[j] = (transpose(sin_mat[j]) * cos_mat[j]) * normalization;
  }
}

// =====================================================================
// Parameters Block (Unchanged)
// =====================================================================
// These are the parameters that will be sampled from their priors.
parameters {
  real tau_logit;
  real delta_logit;
  real lambda_log;
  real phi_log;
  real alpha_r_logit;
  real beta_r_logit;
  real c_r_logit;
  real alpha_s_logit;
  real beta_s_logit;
  real c_s_logit;
  vector[2] y_base_log;
  vector[2] y_pert_log;
  real c_c_logit;
  array[3] real<lower=0> alpha_gp;
  array[3] real<lower=0> rho_gp;
  array[3] vector[N_sin] z_gp;
  array[3] vector[N_sin] z_sin;
  array[3] vector[N_sin] z_cos;
  real w_logit;
}

// =====================================================================
// Transformed Parameters Block (Empty)
// =====================================================================
// This block is left empty. All calculations are moved to `generated quantities`.
transformed parameters {
}

// =====================================================================
// Model Block (Modified)
// =====================================================================
// Note: The likelihood `RR ~ ...` is GONE. This block ONLY specifies the priors.
model {
  // === Priors ===
  tau_logit ~ normal(logit((tau_mu - t_min) / t_range), 0.2);
  delta_logit ~ normal(logit(delta_mu / (t_range - tau_mu)), 0.2);
  lambda_log ~ normal(log(lambda_mu), 0.2);
  phi_log ~ normal(log(phi_mu), 0.2);

  alpha_r_logit ~ normal(0, 2);
  beta_r_logit  ~ normal(0, 2);
  c_r_logit     ~ normal(0, 2);
  alpha_s_logit ~ normal(0, 2);
  beta_s_logit  ~ normal(0, 2);
  c_s_logit     ~ normal(0, 2);

  y_base_log ~ normal([0, 0]', 2);
  y_pert_log ~ normal([0, 0]', 2);
  c_c_logit ~ normal(1, 2);

  alpha_gp ~ normal(0, 0.5) T[0, ];
  rho_gp   ~ lognormal(0, 0.5);

  for (j in 1:3) {
    z_gp[j] ~ std_normal();
    z_sin[j] ~ std_normal();
    z_cos[j] ~ std_normal();
  }

  w_logit ~ normal(3, 2);
}

// =====================================================================
// Generated Quantities Block (New)
// =====================================================================
// This block now contains all the logic to build a synthetic RRi time series
// from a single draw of the parameters from their priors.
generated quantities {
  // All variables previously in `transformed parameters` are now declared here.
  vector[N] mu;
  vector[N] var_resid;
  vector[N] RR_prior; // The final output: a synthetic RRi time series.

  // --- 0. Map unconstrained parameters to their meaningful scales ---
  real tau    = inv_logit(tau_logit) * t_range + t_min;
  real delta  = inv_logit(delta_logit) * (t_range - tau);
  real lambda = exp(lambda_log);
  real phi    = exp(phi_log);

  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range_hypothetical + rr_min_hypothetical;
  real beta_r  = inv_logit(beta_r_logit) * alpha_r;
  real c_r     = inv_logit(c_r_logit) * 2;

  real alpha_s = inv_logit(alpha_s_logit) * rr_sd_hypothetical;
  real beta_s  = inv_logit(beta_s_logit) * alpha_s;
  real c_s     = inv_logit(c_s_logit) * 2;

  real c_c = inv_logit(c_c_logit);
  real w   = inv_logit(w_logit);

  // --- 1. Construct the two logistic building blocks ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Construct the baseline and SDNN trajectories ---
  vector[N] RR_baseline = alpha_r - beta_r .* D1 + (c_r * beta_r) .* D2;
  vector[N] SDNN_t      = alpha_s - beta_s .* D1 + (c_s * beta_s) .* D2;

  // --- 3. Construct the master controller C(t) and spectral proportions p_t ---
  vector[N] C_t = D1 .* (1.0 - c_c .* D2);
  vector[3] pi_base = softmax(append_row(y_base_log, 0.0));
  vector[3] pi_pert = softmax(append_row(y_pert_log, 0.0));
  matrix[N, 3] p_t  = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 4. Construct the spectral coefficient vectors (u_sin, u_cos) ---
  array[3] vector[N_sin] u_sin;
  array[3] vector[N_sin] u_cos;
  array[3] vector[N_sin] log_v;
  for (j in 1:3) {
    matrix[N_sin, N_sin] K =
      gp_exp_quad_cov(to_array_1d(log_freqs[j]), alpha_gp[j], rho_gp[j])
      + diag_matrix(rep_vector(1e-8 * square(alpha_gp[j]) + 1e-12, N_sin));
    matrix[N_sin, N_sin] L = cholesky_decompose(K);
    log_v[j] = L * z_gp[j];
    vector[N_sin] a_k = exp(log_v[j]);
    vector[N_sin] diag_sum = diagonal(G_sin[j]) + diagonal(G_cos[j]);
    real base_v_diag = dot_product(a_k .* a_k, diag_sum);
    vector[N_sin] full_scale = a_k / sqrt(base_v_diag + 1e-12);
    u_sin[j] = z_sin[j] .* full_scale;
    u_cos[j] = z_cos[j] .* full_scale;
  }

  // --- 5. Synthesize the raw oscillator signals and mean-center them ---
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    vector[N] S_j = sin_mat[j] * u_sin[j] + cos_mat[j] * u_cos[j];
    S_t_matrix[:, j] = S_j - mean(S_j);
  }

  // --- 6. Calculate the exact variance of each oscillator using Gram matrices ---
  matrix[3, 3] Sigma_S = rep_matrix(0.0, 3, 3);
  for (j in 1:3) {
    real vj = dot_product(u_sin[j], G_sin[j] * u_sin[j])
              + dot_product(u_cos[j], G_cos[j] * u_cos[j])
              + 2 * dot_product(u_sin[j], G_sin_cos[j] * u_sin[j]);
    Sigma_S[j, j] = vj;
  }

  // --- 7. Deterministic Inversion to find the amplitude A(t) ---
  vector[N] var_struct = square(SDNN_t) .* w;
  matrix[N, 3] M = p_t * Sigma_S;
  vector[N] denom_sq = rows_dot_product(M, p_t);
  vector[N] denom = sqrt(denom_sq + 1e-12);
  vector[N] A_t = sqrt(var_struct) ./ denom;

  // --- 8. Combine components to get the final predicted mean `mu` ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  mu = RR_baseline + A_t .* sum_weighted_S;

  // --- 9. Define the residual variance ---
  var_resid = square(SDNN_t) .* (1.0 - w);

  // --- 10. Generate the final synthetic RRi time series ---
  // For each posterior draw, generate a single realization of the time series.
  for (n in 1:N) {
    RR_prior[n] = normal_rng(mu[n], sqrt(var_resid[n] + 1e-12));
  }
}
