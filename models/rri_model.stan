// Full Stan model: Generative RRi with double-logistic mean and SDNN,
// multiplicative master controller, deterministic amplitude inversion,
// multi-sine spectral texture (phases fixed), and combined structured + obs noise.

functions {
  // Vectorized logistic curve (sigmoid)
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

parameters {
  // shared timing/rate for double-logistics
  real<lower=0> tau;
  real<lower=0> delta;
  real<lower=0> lambda;
  real<lower=0> phi;

  // baseline RR(t) params
  real<lower=0> alpha_r;
  real<lower=0, upper=alpha_r> beta_r;
  real<lower=0, upper=2> c_r;

  // SDNN(t) params (structured variability template)
  real<lower=0> alpha_s;
  real<lower=0, upper=alpha_s> beta_s;
  real<lower=0, upper=2> c_s;

  // Within band noise structure
  real<lower=0> b;

  // spectral proportions endpoints and controller
  simplex[3] pi_base;
  simplex[3] pi_pert;
  real<lower=0, upper=1> c_c;

  // observation / unstructured noise
  real<lower=0> sigma;
}

transformed parameters {
  // epsilon for numerical guards
  real eps = 1e-6;

  // --- 1) logistic building blocks
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2) baseline mean and SDNN trajectories (double-logistic)
  vector[N] RR_baseline = alpha_r - beta_r .* D1 + (c_r * beta_r) .* D2;
  vector[N] SDNN_t      = alpha_s - beta_s .* D1 + (c_s * beta_s) .* D2;

  // --- 3) multiplicative master controller (guaranteed in [0,1])
  vector[N] C_t = D1 .* (1.0 - c_c .* D2);

  // --- 4) proportions p_t as vectorized convex combination (rowwise simplex)
  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 5) spectral synthesis: one normalized narrow-band texture per band
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    // amplitude law a_k = freqs^{-1/2} (vectorized, stable)
    vector[N_sin] a_k = exp(-0.5 * b * log(freqs[j])); // elementwise

    // build matrix of (t_seconds * freq_k) + phase
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]';  // t in minutes -> seconds
    matrix[N, N_sin] P_mat = rep_matrix(phases[j]', N);

    // evaluate weighted sum of minor sinusoids -> vector[N]
    vector[N] S_j_unnorm = sin(2 * pi() * (T_mat + P_mat)) * a_k;

    // empirical standardization (zero-mean, unit-sd)
    real m = mean(S_j_unnorm);
    real s = sd(S_j_unnorm);
    S_t_matrix[:, j] = (S_j_unnorm - m) / s;
  }

  // --- 6) deterministic inversion for A_t with numeric guard
  // denom = sqrt(sum_j p_j(t)^2), clipped to eps
  vector[N] denom = fmax(sqrt(rows_dot_self(p_t)), eps);
  vector[N] A_t = SDNN_t ./ denom;

  // --- 7) predicted mean and structured variance
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t); // elementwise row dot
  vector[N] mu = RR_baseline + A_t .* sum_weighted_S;

  // because A_t was deterministically computed from SDNN_t, structured var = SDNN_t^2
  vector[N] var_struct = square(SDNN_t);
}

model {
  // --- Priors (weakly informative; tune for your domain)
  tau ~ normal(6, 2) T[0, ];
  delta ~ normal(3, 1) T[0, ];
  lambda ~ normal(3, 1) T[0, ];
  phi ~ normal(2, 1) T[0, ];

  alpha_r ~ normal(800, 50) T[0, ];
  beta_r ~ normal(400, 50) T[0, alpha_r];
  c_r ~ normal(0.8, 0.4) T[0, 2];

  alpha_s ~ normal(50, 10) T[0, ];
  beta_s ~ normal(20, 10) T[0, alpha_s];
  c_s ~ normal(1.2, 0.5) T[0, 2];

  b ~ normal(1, 0.1)T[0, ];

  // Dirichlet concentration vectors (concise literal)
  vector[3] alpha_base = [2, 6, 12]';
  vector[3] alpha_pert = [14, 4, 2]';
  pi_base ~ dirichlet(alpha_base);
  pi_pert ~ dirichlet(alpha_pert);

  c_c ~ beta(12, 2);            // informative toward higher recovery proportion (optional)
  sigma ~ normal(0, 1) T[0, ];

  // --- Likelihood: observation SD = sqrt(sigma^2 + var_struct)
  // Guard the stddev to avoid exact zeros (small numerical eps)
  vector[N] sd_obs = fmax(sqrt(square(sigma) + var_struct), 1e-8);
  RR ~ normal(mu, sd_obs);
}
