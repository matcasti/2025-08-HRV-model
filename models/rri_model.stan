// Stan Implementation of the Full Generative R-R Interval Model
//
// This model deconstructs an R-R interval time series into several components:
// 1. A time-varying baseline (mean R-R interval).
// 2. A time-varying total standard deviation (SDNN).
// 3. A structured, oscillatory signal representing physiological variability.
// 4. An unstructured residual noise component.
// The model uses a double-logistic form to capture a perturbation-recovery dynamic.
//
// Final Version Features:
// - Mechanistic double-logistic core for dynamics.
// - Non-parametric Gaussian Process (GP) prior for spectral shape.
// - Non-Centered Parameterization (NCP) for the GP and oscillator coefficients.
// - Deterministic inversion to solve for time-varying amplitude.
// - Exact variance calculation using pre-computed Gram matrices.

// =====================================================================
// Functions Block
// =====================================================================
// This block defines custom functions that can be used elsewhere in the Stan program.
// Defining functions here is efficient, especially for repeated calculations.
functions {
  /**
   * Calculates a standard logistic growth curve (sigmoid).
   * This function is a fundamental building block for modeling dynamic changes over time.
   * @param t The time vector.
   * @param location The time point of the curve's inflection point (midpoint).
   * @param rate The steepness or rate of the transition.
   * @return A vector of values between 0 and 1 representing the logistic curve.
   */
  vector logistic_curve(vector t, real location, real rate) {
    return inv_logit(rate * (t - location));
  }
}

// =====================================================================
// Data Block
// =====================================================================
// This block declares all the data that will be passed into the model from
// an external source (e.g., R or Python).
data {
  // --- Observed Data ---
  int<lower=1> N;              // Number of data points in the time series.
  vector[N] t;                 // Time vector (e.g., in minutes).
  vector[N] RR;                // Observed R-R intervals (e.g., in ms).

  // --- Fixed Spectral Components (treated as data) ---
  // The model uses a fixed basis of sine and cosine waves.
  // The frequencies are provided, and the model estimates their amplitudes.
  int<lower=1> N_sin;          // Number of sinusoids per frequency band.
  array[3] vector[N_sin] freqs;  // Pre-calculated frequencies for VLF, LF, and HF bands.

  // --- Informative Priors on Double-Logistic (DL) parameters ---
  // Passing these as data allows for setting subject-specific or experiment-specific
  // priors, which can greatly improve model identifiability.
  real<lower=0> tau_mu;        // Expected time of the perturbation event.
  real<lower=0> delta_mu;      // Expected delay between perturbation and recovery.
  real<lower=0> lambda_mu;     // Expected rate of the perturbation.
  real<lower=0> phi_mu;        // Expected rate of the recovery.
}

// =====================================================================
// Transformed Data Block
// =====================================================================
// This block pre-computes quantities that depend only on the data.
// Performing these calculations here is highly efficient, as they are done
// only once before sampling begins, rather than on every iteration.
transformed data {
  // === Data-derived scaling parameters ===
  // These are used to scale parameters to a more natural range (e.g., a standard
  // normal scale), which makes defining priors and sampling more robust.
  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // --- Precompute sine and cosine basis function templates ---
  // These matrices hold the values of sin(2*pi*f*t) and cos(2*pi*f*t) for all
  // time points and all basis frequencies. This is a major performance optimization.
  array[3] matrix[N, N_sin] sin_mat;
  array[3] matrix[N, N_sin] cos_mat;
  for (j in 1:3) {
    // Convert time from minutes to seconds for frequency calculation (Hz).
    matrix[N, N_sin] T_mat = (t * 60) * freqs[j]';
    sin_mat[j] = sin(2 * pi() * T_mat);
    cos_mat[j] = cos(2 * pi() * T_mat);
  }

  // --- Precompute log-frequencies ---
  // The GP prior operates on the log of the frequencies to better handle
  // the wide range of frequencies often seen in HRV.
  array[3] vector[N_sin] log_freqs;
  for (j in 1:3) log_freqs[j] = log(freqs[j]);

  // --- Precompute per-band Gram matrices ---
  // These matrices contain the inner products of the basis vectors (e.g., sin_i * sin_j).
  // They are used later to calculate the *exact* sample variance of the synthesized
  // oscillatory signals, without making the simplifying (and often incorrect)
  // assumption that the basis functions are perfectly orthogonal over the discrete
  // time interval.
  array[3] matrix[N_sin, N_sin] G_sin;
  array[3] matrix[N_sin, N_sin] G_cos;
  array[3] matrix[N_sin, N_sin] G_sin_cos; // For cross-terms between sin and cos.

  // Normalization factor for sample variance (1 / (N-1)).
  real normalization = 1.0 / (N - 1);

  for (j in 1:3) {
    G_sin[j]     = (transpose(sin_mat[j]) * sin_mat[j]) * normalization;
    G_cos[j]     = (transpose(cos_mat[j]) * cos_mat[j]) * normalization;
    G_sin_cos[j] = (transpose(sin_mat[j]) * cos_mat[j]) * normalization;
  }
}

// =====================================================================
// Parameters Block
// =====================================================================
// This block declares all the parameters the model will estimate.
// For optimal performance with Stan's HMC sampler, all parameters are declared on
// an unconstrained scale (the real line, -inf to +inf). They are transformed to their
// meaningful, constrained scales in the `transformed parameters` block.
parameters {
  // --- Shared timing/rate parameters for the double-logistic curves ---
  real tau_logit;        // Location of the perturbation event (logit scale).
  real delta_logit;      // Delay between perturbation and recovery (logit scale).
  real lambda_log;       // Rate of the perturbation (log scale).
  real phi_log;          // Rate of the recovery (log scale).

  // --- Baseline RR(t) trend parameters ---
  real alpha_r_logit;    // Initial baseline RR value (logit-scaled).
  real beta_r_logit;     // Magnitude of the RR drop (logit-scaled).
  real c_r_logit;        // Fractional recovery of the RR drop (logit-scaled).

  // --- Total Variability SDNN(t) trend parameters ---
  real alpha_s_logit;    // Initial baseline SDNN value (logit-scaled).
  real beta_s_logit;     // Magnitude of the SDNN drop (logit-scaled).
  real c_s_logit;        // Fractional recovery of the SDNN drop (logit-scaled).

  // --- Parameters for time-varying frequency proportions p(t) ---
  // These control the transition between a baseline and a perturbed spectral state.
  vector[2] y_base_log;  // Unconstrained (ALR) coords for baseline proportions.
  vector[2] y_pert_log;  // Unconstrained (ALR) coords for perturbed proportions.
  real c_c_logit;        // Fractional recovery of spectral proportions (logit-scaled).

  // --- GP Hyperparameters ---
  // These control the properties of the smooth function for the spectral shape.
  array[3] real<lower=0> alpha_gp; // Marginal SD: Controls the vertical variation of the spectrum.
  array[3] real<lower=0> rho_gp;   // Length-scale: Controls the smoothness of the spectrum.

  // --- Non-Centered GP Parameter ---
  // Instead of sampling the GP function values directly, we sample standard normal
  // deviates and reconstruct the GP. This is the Non-Centered Parameterization (NCP).
  array[3] vector[N_sin] z_gp;     // Standard normal deviates for the GP.

  // --- Non-Centered Oscillator Coefficients ---
  // The final amplitudes for each sine/cosine pair are also non-centered.
  // We estimate standard normal deviates (`z`), which are then scaled.
  array[3] vector[N_sin] z_sin;    // Standard normal deviates for sine coefficients.
  array[3] vector[N_sin] z_cos;    // Standard normal deviates for cosine coefficients.

  // --- Fractional split of SDNN variance ---
  real w_logit;          // Proportion of total variance that is "structured" (logit scale).
}

// =====================================================================
// Transformed Parameters Block
// =====================================================================
// This block constructs the full generative model step-by-step from the parameters.
// All calculations here are performed at every leapfrog step of the sampler.
transformed parameters {
  // Declare variables that will be used in the likelihood calculation.
  vector[N] mu;          // The final predicted mean RRi trajectory.
  vector[N] var_resid;   // The final residual variance trajectory.

  // --- 0. Map unconstrained parameters to their meaningful scales ---
  // This section applies inverse transformations (e.g., inv_logit, exp) to the
  // parameters, mapping them from the unconstrained real line back to their
  // natural, interpretable scales (e.g., time, ms, proportions).
  real tau    = inv_logit(tau_logit) * t_range + t_min;
  real delta  = inv_logit(delta_logit) * (t_range - tau); // Delta is a fraction of remaining time.
  real lambda = exp(lambda_log);
  real phi    = exp(phi_log);

  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min;
  real beta_r  = inv_logit(beta_r_logit) * alpha_r; // Beta is a fraction of Alpha.
  real c_r     = inv_logit(c_r_logit) * 2; // Recovery can be up to 200% (overshoot).

  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;
  real beta_s  = inv_logit(beta_s_logit) * alpha_s; // Beta is a fraction of Alpha.
  real c_s     = inv_logit(c_s_logit) * 2; // Recovery can be up to 200% (overshoot).

  real c_c = inv_logit(c_c_logit); // Spectral recovery is between 0-100%.
  real w   = inv_logit(w_logit);   // Structured variance fraction is between 0-1.

  // --- 1. Construct the two logistic building blocks ---
  // These shared curves drive all the dynamic changes in the model.
  vector[N] D1 = logistic_curve(t, tau, lambda);      // Primary perturbation curve.
  vector[N] D2 = logistic_curve(t, tau + delta, phi); // Recovery curve.

  // --- 2. Construct the baseline and SDNN trajectories ---
  // These define the time-varying mean and total standard deviation of the RR signal.
  vector[N] RR_baseline = alpha_r - beta_r .* D1 + (c_r * beta_r) .* D2;
  vector[N] SDNN_t      = alpha_s - beta_s .* D1 + (c_s * beta_s) .* D2;

  // --- 3. Construct the master controller C(t) and spectral proportions p_t ---
  // C(t) governs the transition from a baseline spectral state to a perturbed one.
  vector[N] C_t = D1 .* (1.0 - c_c .* D2);

  // Convert the unconstrained `y_*_log` parameters to 3-element proportion vectors
  // using the softmax function (an inverse Additive Log-Ratio transform).
  vector[3] pi_base = softmax(append_row(y_base_log, 0.0)); // Baseline state.
  vector[3] pi_pert = softmax(append_row(y_pert_log, 0.0)); // Perturbed state.

  // `p_t` smoothly interpolates between the two spectral states based on C(t).
  matrix[N, 3] p_t  = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- 4. Construct the spectral coefficient vectors (u_sin, u_cos) ---
  // This section combines the GP reconstruction and the final coefficient scaling
  // in a single, efficient loop.
  array[3] vector[N_sin] u_sin;
  array[3] vector[N_sin] u_cos;
  array[3] vector[N_sin] log_v; // This will hold the centered GP values.

  for (j in 1:3) {
    // --- Step A: Reconstruct the Centered GP (Non-Centered Parameterization) ---
    // 1. Construct the covariance matrix K from the hyperparameters.
    matrix[N_sin, N_sin] K =
      gp_exp_quad_cov(to_array_1d(log_freqs[j]), alpha_gp[j], rho_gp[j])
      + diag_matrix(rep_vector(1e-8 * square(alpha_gp[j]) + 1e-12, N_sin));

    // 2. Compute the Cholesky factor of K.
    matrix[N_sin, N_sin] L = cholesky_decompose(K);

    // 3. Construct the centered log_v by transforming the standard normal `z_gp`.
    log_v[j] = L * z_gp[j];

    // 4. Build positive amplitudes from GP
    vector[N_sin] a_k = exp(log_v[j]);

    // expected diagonal contribution to variance under z ~ N(0,1)
    vector[N_sin] diag_sum = diagonal(G_sin[j]) + diagonal(G_cos[j]); // length N_sin

    // expected base variance under N(0,1) coefficients (cross-terms vanish)
    real base_v_diag = dot_product(a_k .* a_k, diag_sum);

    // normalize so expected variance ≈ 1 (avoid divide by zero)
    vector[N_sin] full_scale = a_k / sqrt(base_v_diag + 1e-12);

    // final NCP coefficients
    u_sin[j] = z_sin[j] .* full_scale;
    u_cos[j] = z_cos[j] .* full_scale;
  }

  // --- 5. Synthesize the raw oscillator signals and mean-center them ---
  matrix[N, 3] S_t_matrix;
  for (j in 1:3) {
    vector[N] S_j = sin_mat[j] * u_sin[j] + cos_mat[j] * u_cos[j];
    // Mean-center to ensure the oscillators represent pure variability, not a baseline shift.
    S_t_matrix[:, j] = S_j - mean(S_j);
  }

  // --- 6. Calculate the exact variance of each oscillator using Gram matrices ---
  // This quadratic form calculates Var(S_j) = u' * G * u efficiently and exactly.
  matrix[3, 3] Sigma_S = rep_matrix(0.0, 3, 3);
  for (j in 1:3) {
    real vj = dot_product(u_sin[j], G_sin[j] * u_sin[j])
              + dot_product(u_cos[j], G_cos[j] * u_cos[j])
              + 2 * dot_product(u_sin[j], G_sin_cos[j] * u_cos[j]);
    Sigma_S[j, j] = vj;
  }

  // --- 7. Deterministic Inversion to find the amplitude A(t) ---
  // This is a powerful trick. Instead of trying to sample A(t), we solve for the
  // exact amplitude needed to make the variance of the structured signal match the
  // target structural variance defined by `SDNN_t` and `w`. This improves efficiency.
  vector[N] var_struct = square(SDNN_t) .* w; // Target structured variance.

  // The denominator is the variance of the weighted sum of oscillators if A_t were 1.
  matrix[N, 3] M = p_t * Sigma_S;
  vector[N] denom_sq = rows_dot_product(M, p_t);
  vector[N] denom = sqrt(denom_sq + 1e-12);

  // The final amplitude is the ratio of the target SD to the current SD.
  vector[N] A_t = sqrt(var_struct) ./ denom;

  // --- 8. Combine components to get the final predicted mean `mu` ---
  vector[N] sum_weighted_S = rows_dot_product(S_t_matrix, p_t);
  mu = RR_baseline + A_t .* sum_weighted_S;

  // --- 9. Define the residual variance used in the likelihood ---
  // The variance not explained by the structured oscillators becomes the residual variance.
  var_resid = square(SDNN_t) .* (1.0 - w);
}

// =====================================================================
// Model Block
// =====================================================================
// This block specifies the priors for the parameters and the likelihood function
// that connects the model to the observed data.
model {
  // === Priors ===
  // Priors are placed on the unconstrained parameters. This is essential for
  // ensuring the priors cover the entire space the sampler is exploring.

  // --- Priors for logistic components (timing and rates) ---
  tau_logit ~ normal(logit((tau_mu - t_min) / t_range), 0.1);
  delta_logit ~ normal(logit(delta_mu / (t_range - tau_mu)), 0.1);
  lambda_log ~ normal(log(lambda_mu), 0.1);
  phi_log ~ normal(log(phi_mu), 0.1);

  // --- Priors for RR(t) and SDNN(t) parameters ---
  alpha_r_logit ~ normal(0, 1);
  beta_r_logit  ~ normal(0, 1);
  c_r_logit     ~ normal(0, 1);
  alpha_s_logit ~ normal(0, 1);
  beta_s_logit  ~ normal(0, 1);
  c_s_logit     ~ normal(0, 1);

  // --- Priors for spectral proportion p_j(t) parameters ---
  // These priors encode a physiological hypothesis:
  // Baseline state is HF-dominant (low y_base_log).
  // Perturbed state is LF/VLF-dominant (high y_pert_log).
  y_base_log ~ normal([-1, -1]', 1);
  y_pert_log ~ normal([1, 1]', 1);
  c_c_logit ~ normal(1, 1);

  // --- Priors for GP hyperparameters ---
  // `rho_gp` is given a lognormal prior as it must be positive and often
  // spans orders of magnitude, making its log a natural scale to work on.
  alpha_gp ~ normal(0, 0.5) T[0, ];   // half-normal, favors modest envelope
  rho_gp   ~ lognormal(0, 0.5);

  // --- Priors for Non-Centered Parameters ---
  // As part of the NCP pattern, the "raw" parameters are given standard normal
  // priors. This is the core of what makes the NCP work.
  for (j in 1:3) {
    z_gp[j] ~ std_normal();  // Prior for the non-centered GP.
    z_sin[j] ~ std_normal(); // Prior for the sine coefficients.
    z_cos[j] ~ std_normal(); // Prior for the cosine coefficients.
  }

  // --- Prior for the fractional variance split ---
  w_logit ~ normal(1, 1); // Weakly favors a higher proportion of structured variance.

  // === Likelihood ===
  // This is the core statement that connects the model's predictions (`mu` and
  // `var_resid`) to the actual observed data `RR`. The model learns by trying
  // to find parameter values that make the observed data most plausible under a
  // Normal distribution with a time-varying mean and variance.
  RR ~ normal(mu, sqrt(var_resid));
}
