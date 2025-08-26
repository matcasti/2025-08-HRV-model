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
    return 1.0 ./ (1.0 + exp(-rate * (t - location)));
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

parameters {
  // === Shared Timing & Rate Parameters for the Double Logistic functions ===
  real tau;              // Inflection time of the initial drop (shared)
  real<lower=0> delta;   // Temporal offset for recovery start (t_recovery = tau + delta)
  real<lower=0> lambda;  // Rate constant for the initial drop (shared)
  real<lower=0> phi;     // Rate constant for the recovery (shared)

  // === Parameters for Baseline Heart Period: RR(t) ===
  real<lower=0> alpha_r; // Resting RRi
  real<lower=0> beta_r;  // Magnitude of exercise-induced RRi drop
  real<lower=0, upper=2> c_r; // Fractional recovery of RRi (can overshoot)

  // === Parameters for Structured Variability: SDNN(t) ===
  real<lower=0> alpha_s; // Resting SDNN
  real<lower=0> beta_s;  // Magnitude of exercise-induced SDNN drop
  real<lower=0, upper=2> c_s; // Fractional recovery of SDNN

  // === Parameters for Frequency Proportions: p(t) ===
  simplex[3] pi_base;    // Proportions at baseline [VLF, LF, HF]
  simplex[3] pi_pert;    // Proportions during perturbation
  real<lower=0, upper=1> c_c; // Proportional recovery of the master controller C(t)

  // === Parameter for Spectral Content: S(t) ===
  real<lower=0> b;       // Spectral exponent (beta in previous discussions)

  // === Error Term ===
  real<lower=0> sigma;   // Residual, unstructured variability
}

transformed parameters {
  // This block deterministically builds the predicted signal 'mu' from the parameters.

  // --- 1. Define the shared logistic components D1 and D2 ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- 2. Build the baseline heart period trajectory: RR(t) ---
  vector[N] RR_baseline = alpha_r - beta_r * D1 + c_r * beta_r * D2;

  // --- 3. Build the target SDNN trajectory: SDNN(t) ---
  // Clamp at a small positive number to ensure numerical stability.
  vector[N] SDNN_t = fmax(1e-6, alpha_s - beta_s * D1 + c_s * beta_s * D2);

  // --- 4. Build the master controller C(t) and proportions p(t) ---
  vector[N] C_t_unclamped = D1 - c_c * D2;
  // Clamp C(t) to the [0, 1] range to ensure valid proportions.
  vector[N] C_t = fmax(0.0, fmin(1.0, C_t_unclamped));

  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 5. Build the spectral oscillators S_j(t) ---
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    vector[N_sin] a_k = freqs[j] .^ (-b / 2.0);
    // Transpose `t` to a row_vector to create a matrix `T_mat` that we can add `P_mat` to.
    matrix[N, N_sin] T_mat = t * freqs[j]';
    matrix[N, N_sin] P_mat = rep_matrix(phases[j]', N);
    vector[N] S_j_unnormalized = (sin(2 * pi() * (T_mat + P_mat))) * a_k;
    S_t_matrix[:, j] = (S_j_unnormalized - mean(S_j_unnormalized)) / sd(S_j_unnormalized);
  }

  // --- 6. Derive the internal amplitude A(t) using the corrected inversion ---
  // rows_dot_self(p_t) efficiently calculates sum(p_j^2) for each time point.
  vector[N] sum_p_sq = rows_dot_self(p_t);
  vector[N] A_t = SDNN_t ./ sqrt(sum_p_sq);

  // --- 7. Combine for the final predicted signal mu ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  vector[N] mu = RR_baseline + A_t .* sum_weighted_S;
}

model {
  // === Priors (Weakly informative priors for all parameters) ===
  // Timing
  tau ~ normal(6, 0.1) T[0, ];
  delta ~ normal(3, 0.1) T[0, ]; // Recovery likely starts after a few mins
  lambda ~ normal(3, 0.1) T[0, ];  // Rates are on a log scale
  phi ~ normal(3, 0.1) T[0, ];

  // RR(t) parameters
  alpha_r ~ normal(800, 50) T[0, ];
  beta_r ~ normal(400, 50) T[0, ];
  c_r ~ normal(0.8, 0.1) T[0, 2];

  // SDNN(t) parameters
  alpha_s ~ normal(50, 5) T[0, ];
  beta_s ~ normal(20, 5) T[0, ];
  c_s ~ normal(1.2, 0.1) T[0, 2];

  // Proportion parameters
  pi_base ~ dirichlet([2,3,6]');
  pi_pert ~ dirichlet([6,3,2]');
  c_c ~ beta(4, 1); // Prior belief in fairly complete recovery

  // Spectral parameter
  b ~ normal(0, 1) T[0, ]; // Prior centered around b=1 (pink noise)

  // Error term
  sigma ~ normal(0, 5) T[0, ];

  // === Likelihood ===
  RR ~ normal(mu, sigma);
}
