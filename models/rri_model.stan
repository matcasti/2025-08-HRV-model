// Stan Implementation of the Full Generative R-R Interval Model

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
  int<lower=1> N;           // Number of data points
  vector[N] t;              // Time vector (in seconds)
  vector[N] RR;             // Observed R-R intervals (in ms)

  // --- Fixed Spectral Components (treated as data) ---
  int<lower=1> N_sin;       // Number of sinusoids per band
  array[3] vector[N_sin] freqs;  // Pre-calculated frequencies for VLF, LF, HF
  array[3] vector[N_sin] phases; // Pre-calculated phases for VLF, LF, HF
}

transformed data {
  // Data derived parameters
  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // Precompute sin templates (depends only on data: t, freqs, phases)
  array[3] matrix[N, N_sin] sin_mat;
  for (j in 1:3) {
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]'; // t in minutes -> seconds
    matrix[N, N_sin] P_mat = rep_matrix(phases[j]', N);
    sin_mat[j] = sin(2 * pi() * (T_mat + P_mat));
  }

  // Precompute log_freqs to avoid recomputing the log at each iteration
  array[3] vector[N_sin] log_freqs;
  for (j in 1:3) log_freqs[j] = log(freqs[j]);
}

parameters {
  // === Unconstrained parameters in the logit/log scale === //

  // Shared timing/rate for double-logistics
  real tau_logit;
  real delta_logit;
  real lambda_log;
  real phi_log;

  // Baseline RR(t) params
  real alpha_r_logit;
  real beta_r_logit;
  real c_r_logit;
  // SDNN(t) params
  real alpha_s_logit;
  real beta_s_logit;
  real c_s_logit;

  // === Parameters for Frequency Proportions: p(t) ===
  simplex[3] pi_base;    // Proportions at baseline [VLF, LF, HF]
  simplex[3] pi_pert;    // Proportions during perturbation
  real c_c_logit;        // Proportional recovery of the master controller C(t)

  // Within band noise structure
  real b_log;

  // === Error Term ===
  // real<lower=0> sigma;   // Residual, unstructured variability
}

transformed parameters {
  // === Computing constrained from the unconstrained parameters === //
  // Shared timing/rate for double-logistics
  real tau = inv_logit(tau_logit) * t_range + t_min; // [t_min, t_min + t_range]
  real delta = inv_logit(delta_logit) * t_range;     // [0, t_range]
  real lambda = exp(lambda_log);                     // [0, Inf]
  real phi = exp(phi_log);                           // [0, Inf]

  // Baseline and Drop params
  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min; // [rr_min, rr_min + 2 * rr_range]
  real beta_r = inv_logit(beta_r_logit) * alpha_r;                 // [0, alpha_r]
  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;                 // [0, rr_sd]
  real beta_s = inv_logit(beta_s_logit) * alpha_s;                 // [0, alpha_s]

  // Recovery params
  real c_r = inv_logit(c_r_logit) * 2; // [0,2]
  real c_s = inv_logit(c_s_logit) * 2; // [0,2]
  real c_c = inv_logit(c_c_logit); // [0,1]

  // Within band noise structure
  real b = exp(b_log); // [0, Inf]

  // --- 1. Define the shared logistic components D1 and D2 ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Build the baseline heart period trajectory: RR(t) ---
  vector[N] RR_baseline = alpha_r - beta_r * D1 + c_r * beta_r * D2;

  // --- 3. Build the target SDNN trajectory: SDNN(t) ---
  // Clamp at a small positive number to ensure numerical stability.
  vector[N] SDNN_t = alpha_s - beta_s * D1 + c_s * beta_s * D2;

  // --- 4. Build the master controller C(t) and proportions p(t) ---
  vector[N] C_t = D1 .* (1 - c_c .* D2);

  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 5. Build the spectral oscillators S_j(t) ---
  // --- spectral synthesis using precomputed sin templates
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    // amplitude law a_k = freqs^{-b/2}
    vector[N_sin] a_k = exp(-b/2 .* log_freqs[j]);

    // build unnormalized band signal via sin_mat (data) * a_k
    vector[N] S_j_unnorm = sin_mat[j] * a_k;

    // empirical standardization with guard
    real s = sd(S_j_unnorm);
    real m = mean(S_j_unnorm);
    S_t_matrix[:, j] = (S_j_unnorm - m) ./ fmax(s, 1e-8);
  }

  // --- 6. Derive the internal amplitude A(t) using inversion ---
  // rows_dot_self(p_t) efficiently calculates sum(p_j^2) for each time point.
  vector[N] sum_p_sq = rows_dot_self(p_t);
  vector[N] A_t = 1 ./ sqrt(sum_p_sq);

  // --- 7. Combine for the final predicted signal mu ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  vector[N] mu = RR_baseline + A_t .* sum_weighted_S;
}

model {
  // --- Priors
  tau_logit ~ normal(-0.4, 0.1);
  delta_logit ~ normal(-1.4, 0.1);
  lambda_log ~ normal(1.1, 0.1);
  phi_log ~ normal(0.7, 0.1);

  alpha_r_logit ~ normal(-0.4, 0.1);
  beta_r_logit  ~ normal(0, 0.1);
  c_r_logit     ~ normal(-0.4, 0.1);

  alpha_s_logit ~ normal(-0.5, 0.1);
  beta_s_logit  ~ normal(0, 0.1);
  c_s_logit     ~ normal(0.4, 0.1);

  // Spectral parameter
  b_log ~ normal(0, 0.1); // Prior centered around b=1 (pink noise)

  // Proportion parameters
  pi_base ~ dirichlet([10,30,60]');
  pi_pert ~ dirichlet([70,20,10]');
  c_c_logit ~ normal(1.4, 0.1);

  // Error term
  // sigma ~ normal(0, 1) T[0, ];

  // === Likelihood ===
  RR ~ normal(mu, SDNN_t);
}
