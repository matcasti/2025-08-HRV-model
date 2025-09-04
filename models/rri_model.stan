// Stan Implementation of the Full Generative R-R Interval Model
// Adapted to use Sine/Cosine pairs for robust phase/amplitude estimation

functions {
  /**
   * Calculates a standard logistic growth curve (sigmoid).
   * @param t time vector
   * @param location inflection point of the curve (tau or tau + delta)
   * @param rate rate parameter (lambda or phi)
   * @return A vector of values between 0 and 1 representing the logistic curve.
   */
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

  // --- RR and time-specific magnitudes
  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // --- Precompute sin and cos templates (no phase component) ---
  array[3] matrix[N, N_sin] sin_mat;
  array[3] matrix[N, N_sin] cos_mat; // Cosine basis matrix
  for (j in 1:3) {
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]'; // t in minutes -> seconds
    sin_mat[j] = sin(2 * pi() * T_mat);
    cos_mat[j] = cos(2 * pi() * T_mat); // NEW: Precompute cosine basis
  }

  // --- Precompute log_freqs to avoid recomputing the log at each iteration ---
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
  vector[2] y_base_log;      // unconstrained log-ratio for baseline
  vector[2] y_pert_log;      // unconstrained log-ratio for perturbation
  real c_c_logit;            // Proportional recovery of the master controller C(t)

  // --- Spectral parameters ---
  real b_log; // Exponent for 1/f noise structure

  // Standard normal deviates for sine and cosine coefficients
  // This is a non-centered parameterization for the spectral amplitudes
  array[3] vector[N_sin] z_sin;
  array[3] vector[N_sin] z_cos;

  // --- Fractional split of SDNN ---
  real w_logit;
}

transformed parameters {
  // --- 0. Computing constrained from the unconstrained parameters ---
  // Shared timing/rate for double-logistics
  real tau = inv_logit(tau_logit) * t_range + t_min;
  real delta = inv_logit(delta_logit) * (t_range - tau);
  real lambda = exp(lambda_log);
  real phi = exp(phi_log);

  // Baseline and Drop params
  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min;
  real beta_r = inv_logit(beta_r_logit) * alpha_r;
  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;
  real beta_s = inv_logit(beta_s_logit) * alpha_s;

  // Recovery params
  real c_r = inv_logit(c_r_logit) * 2;
  real c_s = inv_logit(c_s_logit) * 2;
  real c_c = inv_logit(c_c_logit);

  // Spectral exponent
  real b = exp(b_log);

  // Fractional split of SDNN
  real w = inv_logit(w_logit);

  // --- 1. Define the shared logistic components D1 and D2 ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Build the baseline heart period trajectory: RR_baseline ---
  vector[N] RR_baseline = alpha_r - beta_r * D1 + c_r * beta_r * D2;

  // --- 3. Build the target SDNN trajectory: SDNN_t ---
  vector[N] SDNN_t = alpha_s - beta_s * D1 + c_s * beta_s * D2;

  // --- 4. Build the master controller C(t) and proportions p(t) ---
  vector[N] C_t = D1 .* (1 - c_c .* D2);

  // Map from ALR coordinates (y) to simplex (pi)
  vector[3] pi_base;
  vector[3] pi_pert;
  { // local block for clarity
    real denom_base = 1 + exp(y_base_log[1]) + exp(y_base_log[2]);
    pi_base[1] = exp(y_base_log[1]) / denom_base;
    pi_base[2] = exp(y_base_log[2]) / denom_base;
    pi_base[3] = 1 / denom_base;

    real denom_pert = 1 + exp(y_pert_log[1]) + exp(y_pert_log[2]);
    pi_pert[1] = exp(y_pert_log[1]) / denom_pert;
    pi_pert[2] = exp(y_pert_log[2]) / denom_pert;
    pi_pert[3] = 1 / denom_pert;
  }
  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 5. Build the spectral oscillators S_j(t) using sine/cosine pairs ---
  matrix[N, 3] S_t_matrix;
  { // local block for clarity
    // Define coefficients for the sine and cosine bases
    array[3] vector[N_sin] beta_sin;
    array[3] vector[N_sin] beta_cos;
    for (j in 1:3) {
      // The amplitude law a_k = freqs^{-b/2} now acts as the scale for the coefficients
      vector[N_sin] a_k = exp(-b/2 .* log_freqs[j]);
      // Scale the standard normal deviates to get the final coefficients
      beta_sin[j] = z_sin[j] .* a_k;
      beta_cos[j] = z_cos[j] .* a_k;
    }

    // Spectral synthesis using precomputed sin and cos templates
    for (j in 1:3) {
      // Build unnormalized band signal from the linear combination of sin and cos bases
      vector[N] S_j_unnorm = sin_mat[j] * beta_sin[j] + cos_mat[j] * beta_cos[j];

      // Empirical mean centering
      real m = mean(S_j_unnorm);
      S_t_matrix[:, j] = (S_j_unnorm - m);
    }
  }

  // Compute 3x3 covariance of the basis signals
  matrix[3,3] Sigma_S = crossprod(S_t_matrix) / (N - 1);

  // --- 6. Derive the internal amplitude A(t) using inversion ---
  vector[N] var_struct = square(SDNN_t) * w;
  vector[N] var_resid = square(SDNN_t) * (1 - w);
  vector[N] A_t;
  { // local block for clarity
    matrix[N,3] M = p_t * Sigma_S;
    vector[N] denom_sq = rows_dot_product(M, p_t);
    // Add a small epsilon to prevent division by zero if a band has no power
    vector[N] denom = sqrt(denom_sq + 1e-9);
    A_t = sqrt(var_struct) ./ denom;
  }

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

  // --- RR(t) parameters ---
  alpha_r_logit ~ normal(0, 1);
  beta_r_logit  ~ normal(0, 1);
  c_r_logit     ~ normal(0, 1);

  // --- SDNN(t) parameters ---
  alpha_s_logit ~ normal(0, 1);
  beta_s_logit  ~ normal(0, 1);
  c_s_logit     ~ normal(0, 1);

  // --- p_j(t) parameters ---
  y_base_log ~ normal([-1, -1]', 1);
  y_pert_log ~ normal([ 1,  1]', 1);
  c_c_logit ~ normal(1, 1);

  // --- Spectral parameters ---
  b_log ~ normal(0, 0.2); // Prior centered around b=1 (pink noise)

  // Priors on the unscaled standard normal coefficients
  for (j in 1:3) {
    z_sin[j] ~ std_normal();
    z_cos[j] ~ std_normal();
  }

  // --- Fractional split of SDNN ---
  w_logit ~ normal(1, 1); // Prior belief of more structured noise

  // === Likelihood ===
  RR ~ normal(mu, sqrt(var_resid));
}
