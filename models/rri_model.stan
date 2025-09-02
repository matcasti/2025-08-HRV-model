
functions {
  // Logistic curve
  vector logistic_curve(vector t, real location, real rate) {
    return inv_logit(rate * (t - location));
  }
}

data {
  int<lower=1> N;                 // number of observations
  vector[N] t;                    // time in minutes
  vector[N] RR;                   // observed R-R intervals (ms)

  int<lower=1> N_sin;             // number of minor sinusoids per band
  array[3] vector[N_sin] freqs;   // frequencies, in Hz (cycles / sec)
  array[3] vector[N_sin] phases;  // phases, radians (fixed, passed as data)
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

  // Concentration factor
  real alpha_factor = 50;
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

  // Within band noise structure
  real<lower=0> b_log;

  // spectral proportions endpoints and controller
  simplex[3] pi_base;
  simplex[3] pi_pert;
  real<lower=0, upper=1> c_c;

  // observation / unstructured noise
  real<lower=0> sigma;
}

transformed parameters {
  // === Computing constrained from the unconstrained parameters === //
  // Shared timing/rate for double-logistics
  real tau = inv_logit(tau_logit) * t_range + t_min; // [t_min, t_min + t_range]
  real delta = inv_logit(delta_logit) * t_range;     // [0, t_range]
  real lambda = exp(lambda_log);                     // [0, Inf]
  real phi = exp(phi_log);                           // [0, Inf]

  // Baseline RR(t) params
  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min; // [rr_min, rr_min + 2 * rr_range]
  real beta_r = inv_logit(beta_r_logit) * alpha_r;                 // [0, alpha_r]
  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;                 // [0, rr_sd]
  real beta_s = inv_logit(beta_s_logit) * alpha_s;                 // [0, alpha_s]

  // SDNN(t) params
  real c_r = inv_logit(c_r_logit) * 2; // [0,2]
  real c_s = inv_logit(c_s_logit) * 2; // [0,2]

  // Within band noise structure
  real b = exp(b_log); // [0, Inf]

  // --- logistic building blocks (depends on parameters)
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- baseline mean and SDNN trajectories (double-logistic)
  vector[N] RR_t   = alpha_r - beta_r .* D1 + (c_r * beta_r) .* D2;
  vector[N] SDNN_t = alpha_s - beta_s .* D1 + (c_s * beta_s) .* D2;

  // --- multiplicative master controller (guaranteed in [0,1])
  vector[N] C_t = D1 .* (1.0 - c_c .* D2);

  // --- proportions p_t as vectorized convex combination (rowwise simplex)
  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

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

  // --- deterministic inversion for A_t with numeric guard
  vector[N] denom = sqrt(rows_dot_self(p_t));
  vector[N] A_t = SDNN_t ./ fmax(denom, 1e-8);

  // --- predicted mean and structured variance
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  vector[N] mu = RR_t + A_t .* sum_weighted_S;

  // --- total variance
  // vector[N] var_total = square(SDNN_t) + square(sigma);
  vector[N] var_total = square(SDNN_t);
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

  b_log ~ normal(0, 0.1);

  // Dirichlet concentration vectors (concise literal scaled by factor)
  vector[3] alpha_base = alpha_factor * [0.1, 0.3, 0.6]';
  vector[3] alpha_pert = alpha_factor * [0.7, 0.2, 0.1]';
  pi_base ~ dirichlet(alpha_base);
  pi_pert ~ dirichlet(alpha_pert);

  c_c ~ beta(16, 4) T[0, 1];

  sigma ~ normal(0, 1) T[0, ];

  // --- Likelihood: observation SD = sqrt(sigma^2 + sdnn^2)
  RR ~ normal(mu, sqrt(var_total));
}
