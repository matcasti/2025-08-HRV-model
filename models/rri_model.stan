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
  vector[N] t;              // Time vector (in minutes)
  vector[N] RR;             // Observed R-R intervals (in ms)

  // --- Fixed Spectral Components (treated as data) ---
  int<lower=1> N_sin;       // Number of sinusoids per band
  array[3] vector[N_sin] freqs;  // Pre-calculated frequencies for VLF, LF, HF
  array[3] vector[N_sin] phases; // Pre-calculated phases for VLF, LF, HF
}

transformed data {
  // === Data derived parameters ===

  // --- RR and time-specific magnitudes
  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // --- Precompute sin templates (depends only on data: t, freqs, phases) ---
  array[3] matrix[N, N_sin] sin_mat;
  for (j in 1:3) {
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]'; // t in minutes -> seconds
    matrix[N, N_sin] P_mat = rep_matrix(phases[j]', N);
    sin_mat[j] = sin(2 * pi() * (T_mat + P_mat));
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
  real c_c_logit;        // Proportional recovery of the master controller C(t)

  // --- Within band noise structure ---
  real b_log;

  // Fractional split of SDNN
  real w_logit;
}

transformed parameters {
  // --- 0. Computing constrained from the unconstrained parameters ---
  // Shared timing/rate for double-logistics
  real tau = inv_logit(tau_logit) * t_range + t_min; // [t_min, t_min + t_range]
  real delta = inv_logit(delta_logit) * (t_range - tau);     // [0, t_range - tau]
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

  // Fractional split of SDNN
  real w = inv_logit(w_logit); // [0,1]

  // --- 1. Define the shared logistic components D1 and D2 ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Build the baseline heart period trajectory: RR(t) ---
  vector[N] RR_baseline = alpha_r - beta_r * D1 + c_r * beta_r * D2;

  // --- 3. Build the target SDNN trajectory: SDNN(t) ---
  vector[N] SDNN_t = alpha_s - beta_s * D1 + c_s * beta_s * D2;

  // --- 4. Build the master controller C(t) and proportions p(t) ---
  vector[N] C_t = D1 .* (1 - c_c .* D2);

  // --- 5. Build the spectral oscillators S_j(t) ---
  // Map from ALR coordinates (y) to simplex (pi)
  vector[3] pi_base;
  vector[3] pi_pert;

  // For base-state vector
  real denom_base = 1 + exp(y_base_log[1]) + exp(y_base_log[2]);
  pi_base[1] = exp(y_base_log[1]) / denom_base;
  pi_base[2] = exp(y_base_log[2]) / denom_base;
  pi_base[3] = 1 / denom_base;

  // For perturbation-state vector
  real denom_pert = 1 + exp(y_pert_log[1]) + exp(y_pert_log[2]);
  pi_pert[1] = exp(y_pert_log[1]) / denom_pert;
  pi_pert[2] = exp(y_pert_log[2]) / denom_pert;
  pi_pert[3] = 1 / denom_pert;

  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // Spectral synthesis using precomputed sin templates
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    // amplitude law a_k = freqs^{-b/2}
    vector[N_sin] a_k = exp(-b/2 .* log_freqs[j]);

    // build unnormalized band signal via sin_mat (data) * a_k
    vector[N] S_j_unnorm = sin_mat[j] * a_k;

    // empirical mean centering
    real m = mean(S_j_unnorm);
    S_t_matrix[:, j] = (S_j_unnorm - m);
  }

  // Compute 3x3 covariance Sigma_S (use N-1 denom)
  matrix[3,3] Sigma_S;
  for (i in 1:3) {
    for (j in 1:3) {
      Sigma_S[i,j] = dot_product(col(S_t_matrix, i), col(S_t_matrix, j)) / (N - 1);
    }
  }

  // --- 6. Derive the internal amplitude A(t) using inversion ---
  // Compute denom_sq vectorized: denom_sq[i] = p_i' * Sigma_S * p_i
  // First compute M = p_t * Sigma_S  -> matrix[N,3]
  matrix[N,3] M = p_t * Sigma_S; // matrix-matrix multiply

  vector[N] denom_sq = rows_dot_product(M, p_t); // vectorized p_i' * (Sigma p_i)
  vector[N] denom = sqrt(denom_sq);

  // Fraction of SDNN correspond to the structured noise
  vector[N] var_struct = square(SDNN_t) * w;

  // Fraction of SDNN correspond to the residual noise
  vector[N] var_resid = square(SDNN_t) * (1 - w);

  vector[N] A_t = sqrt(var_struct) ./ denom;

  // --- 7. Combine for the final predicted signal mu ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  vector[N] mu = RR_baseline + A_t .* sum_weighted_S;
}

model {
  // === Priors ===

  // --- Logistic components ---
  // It appears that the parameters from the logistic components
  // need tight priors to ensure identifiability, otherwise the model
  // experience multimodal posterior geometry and divergent transitions
  tau_logit ~ normal(logit(0.4), 0.1);
  delta_logit ~ normal(logit(0.3), 0.1);
  lambda_log ~ normal(log(3), 0.1);
  phi_log ~ normal(log(2), 0.1);

  // --- RR(t) parameters ---
  alpha_r_logit ~ normal(0, 1); // 50% of RR range
  beta_r_logit  ~ normal(0, 1); // 50% of resting RR
  c_r_logit     ~ normal(0, 1); // 50% of [0,2] = 1 -> complete RR recovery

  // --- SDNN(t) parameters ---
  alpha_s_logit ~ normal(0, 1); // 50% of total RR SD
  beta_s_logit  ~ normal(0, 1); // 50% of resting SDNN
  c_s_logit     ~ normal(0, 1); // 50% of [0,2] = 1 -> complete SDNN recovery

  // --- p_j(t) parameters ---
  // base: VLF = 0.10, LF = 0.30, HF = 0.60
  // pert: VLF = 0.70, LF = 0.20, HF = 0.10
  // Map to ALR: mu = [log(VLF/HF), log(LF/HF)]'
  y_base_log ~ normal([-1, -1]', 1); // More HF dominant state
  y_pert_log ~ normal([ 1,  1]', 1); // More VLF-LF dominant state

  c_c_logit ~ normal(1, 1); // Prior belief of partial recovery (above 50%)

  // --- Spectral parameter ---
  b_log ~ normal(0, 0.2); // Prior centered around b=1 (pink noise)

  // --- Fractional split of SDNN ---
  w_logit ~ normal(1, 1); // Prior belief of more structured noise

  // === Likelihood ===
  RR ~ normal(mu, sqrt(var_resid));
}
