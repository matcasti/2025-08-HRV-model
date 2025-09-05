// Stan Implementation of the Full Generative R-R Interval Model

functions {
  // Calculates a standard logistic growth curve (sigmoid).
  vector logistic_curve(vector t, real location, real rate) {
    return inv_logit(rate * (t - location));
  }
}

data {
  // --- Observed Data ---
  int<lower=1> N;              // Number of data points
  vector[N] t;                 // Time vector (in minutes)
  vector[N] RR;                // Observed R-R intervals (in ms)

  // --- Fixed Spectral Components (treated as data) ---
  int<lower=1> N_sin;          // Number of sinusoids per band
  array[3] vector[N_sin] freqs; // Pre-calculated frequencies for VLF, LF, HF
}

transformed data {
  // === Data derived parameters ===
  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // --- Precompute sin and cos templates ---
  array[3] matrix[N, N_sin] sin_mat;
  array[3] matrix[N, N_sin] cos_mat;
  for (j in 1:3) {
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]'; // t in minutes -> seconds
    sin_mat[j] = sin(2 * pi() * T_mat);
    cos_mat[j] = cos(2 * pi() * T_mat);
  }

  // --- Precompute log_freqs ---
  array[3] vector[N_sin] log_freqs;
  for (j in 1:3) log_freqs[j] = log(freqs[j]);
}

parameters {
  // === Unconstrained parameters in the logit/log scale === //

  // --- Shared timing/rate for double-logistics ---
  real tau_logit;
  real delta_logit;
  real lambda_log;
  real phi_log;

  // --- Baseline RR(t) params ---
  real alpha_r_logit;
  real beta_r_logit;
  real c_r_logit;
  // --- SDNN(t) params ---
  real alpha_s_logit;
  real beta_s_logit;
  real c_s_logit;

  // --- Parameters for Frequency Proportions: p(t) ---
  vector[2] y_base_log;
  vector[2] y_pert_log;
  real c_c_logit;

  // --- Spectral parameters ---
  real b_log; // Exponent for 1/f noise structure

  // Per-band scale parameters to improve posterior geometry
  vector<lower=0>[3] sigma_beta;

  // Standard normal deviates for sine and cosine coefficients (non-centered)
  array[3] vector[N_sin] z_sin;
  array[3] vector[N_sin] z_cos;

  // --- Fractional split of SDNN ---
  real w_logit;
}

transformed parameters {
  // --- 0. Computing constrained from the unconstrained parameters ---
  real tau = inv_logit(tau_logit) * t_range + t_min;
  real delta = inv_logit(delta_logit) * (t_range - tau);
  real lambda = exp(lambda_log);
  real phi = exp(phi_log);
  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min;
  real beta_r = inv_logit(beta_r_logit) * alpha_r;
  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;
  real beta_s = inv_logit(beta_s_logit) * alpha_s;
  real c_r = inv_logit(c_r_logit) * 2;
  real c_s = inv_logit(c_s_logit) * 2;
  real c_c = inv_logit(c_c_logit);
  real b = exp(b_log);
  real w = inv_logit(w_logit);

  // --- 1. Define the shared logistic components ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Build the baseline and SDNN trajectories ---
  vector[N] RR_baseline = alpha_r - beta_r * D1 + c_r * beta_r * D2;
  vector[N] SDNN_t = alpha_s - beta_s * D1 + c_s * beta_s * D2;

  // --- 4. Build the master controller C(t) and proportions p(t) ---
  vector[N] C_t = D1 .* (1 - c_c .* D2);
  matrix[N, 3] p_t;
  vector[3] pi_base;
  vector[3] pi_pert;

  // block for ALR transform
  real denom_base = 1 + exp(y_base_log[1]) + exp(y_base_log[2]);
  pi_base[1] = exp(y_base_log[1]) / denom_base;
  pi_base[2] = exp(y_base_log[2]) / denom_base;
  pi_base[3] = 1 / denom_base;

  real denom_pert = 1 + exp(y_pert_log[1]) + exp(y_pert_log[2]);
  pi_pert[1] = exp(y_pert_log[1]) / denom_pert;
  pi_pert[2] = exp(y_pert_log[2]) / denom_pert;
  pi_pert[3] = 1 / denom_pert;
  p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 5. Build the spectral oscillators S_j(t) ---
  matrix[N, 3] S_t_matrix;

  // block for spectral synthesis
  array[3] vector[N_sin] beta_sin;
  array[3] vector[N_sin] beta_cos;

  for (j in 1:3) {
    // The amplitude law a_k acts as a frequency-dependent prior scale
    vector[N_sin] a_k = exp(-b/2 .* log_freqs[j]);
    // Per-band scale sigma_beta[j] allows each band's total power
    // to be adjusted, improving sampling efficiency.
    beta_sin[j] = z_sin[j] .* sigma_beta[j] .* a_k;
    beta_cos[j] = z_cos[j] .* sigma_beta[j] .* a_k;
  }

  for (j in 1:3) {
    // This is the primary computational bottleneck: N x P matrix-vector products
    vector[N] S_j_unnorm = sin_mat[j] * beta_sin[j] + cos_mat[j] * beta_cos[j];
    S_t_matrix[:, j] = S_j_unnorm - mean(S_j_unnorm);
  }

  // Compute 3x3 covariance of the basis signals
  matrix[3,3] Sigma_S = crossprod(S_t_matrix) / (N - 1);

  // --- 6. Derive the internal amplitude A(t) using inversion ---
  vector[N] var_struct = square(SDNN_t) * w;
  vector[N] var_resid = square(SDNN_t) * (1 - w);
  vector[N] A_t;

  // block for amplitude calculation
  matrix[N,3] M = p_t * Sigma_S;
  vector[N] denom_sq = rows_dot_product(M, p_t);
  vector[N] denom = sqrt(denom_sq);
  A_t = sqrt(var_struct) ./ denom;

  // --- 7. Combine for the final predicted signal mu ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  vector[N] mu = RR_baseline + A_t .* sum_weighted_S;
}

model {
  // === Priors ===

  // --- Logistic components ---
  tau_logit ~ normal(logit(0.4), 0.1);
  delta_logit ~ normal(logit(0.3), 0.1);
  lambda_log ~ normal(log(3), 0.1);
  phi_log ~ normal(log(2), 0.1);

  // --- RR(t) and SDNN(t) parameters ---
  alpha_r_logit ~ normal(0, 1);
  beta_r_logit  ~ normal(0, 1);
  c_r_logit     ~ normal(0, 1);
  alpha_s_logit ~ normal(0, 1);
  beta_s_logit  ~ normal(0, 1);
  c_s_logit     ~ normal(0, 1);

  // --- p_j(t) parameters ---
  y_base_log ~ normal([-1, -1]', 1);
  y_pert_log ~ normal([ 1,  1]', 1);
  c_c_logit ~ normal(1, 1);

  // --- Spectral parameters ---
  b_log ~ normal(0, 0.2);

  // Prior on the per-band scale parameters. A half-normal provides
  // gentle regularization towards zero, allowing bands to have low power.
  sigma_beta ~ normal(0, 0.5);

  // Priors on the unscaled standard normal coefficients
  for (j in 1:3) {
    z_sin[j] ~ std_normal();
    z_cos[j] ~ std_normal();
  }

  // --- Fractional split of SDNN ---
  w_logit ~ normal(1, 1);

  // === Likelihood ===
  RR ~ normal(mu, sqrt(var_resid));
}
