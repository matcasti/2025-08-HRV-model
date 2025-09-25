// Stan Implementation with Generative Filter Bank
// Replaces frequency amplitude estimation with time-domain filtering

functions {
  // Simple IIR band-pass filter (Butterworth-like) implemented in Stan
  vector bandpass_filter(vector x, real f_low, real f_high, real sample_rate, int order) {
    int N = num_elements(x);
    vector[N] y = rep_vector(0.0, N);

    // Normalize frequencies
    real w_low = 2 * pi() * f_low / sample_rate;
    real w_high = 2 * pi() * f_high / sample_rate;

    // Filter coefficients (simplified Butterworth)
    real alpha = (sin(w_high - w_low)) / (sin(w_high) + sin(w_low));
    real beta = cos((w_high + w_low) / 2) / cos((w_high - w_low) / 2);

    // Initialize (assuming initial conditions are zero)
    if (order >= 1) {
      y[1] = alpha * x[1];
      for (n in 2:N) {
        y[n] = alpha * (x[n] - x[n-1]) + beta * y[n-1];
      }
    }

    // Second order if needed
    if (order >= 2) {
      vector[N] y2 = rep_vector(0.0, N);
      y2[1] = alpha * y[1];
      for (n in 2:N) {
        y2[n] = alpha * (y[n] - y[n-1]) + beta * y2[n-1];
      }
      y = y2;
    }

    return y;
  }

  // Alternative: Generate band-limited noise directly
  vector generate_band_limited_noise(int N, real f_low, real f_high, real sample_rate) {
    vector[N] noise = rep_vector(0.0, N);
    int M = 10; // Number of frequency components to approximate band

    for (m in 1:M) {
      real freq = f_low + (f_high - f_low) * (m - 1) / (M - 1);
      real phase = uniform_rng(0, 2 * pi());
      real amp = 1.0 / sqrt(M); // Equal energy per component

      for (n in 1:N) {
        real t = n / sample_rate;
        noise[n] += amp * sin(2 * pi() * freq * t + phase);
      }
    }

    return noise;
  }

  vector logistic_curve(vector t, real location, real rate) {
    return inv_logit(rate * (t - location));
  }
}

data {
  int<lower=1> N;
  vector[N] t;
  vector[N] RR;

  // Priors for double-logistic
  real<lower=0> tau_mu;
  real<lower=0> delta_mu;
  real<lower=0> lambda_mu;
  real<lower=0> phi_mu;
}

transformed data {
  // Filter parameters (fixed based on physiology)
  real vlf_low = 0.003;
  real vlf_high = 0.04;
  real lf_low = 0.04;
  real lf_high = 0.15;
  real hf_low = 0.15;
  real hf_high = 0.4;
  int filter_order = 2;

  real rr_min = min(RR);
  real rr_range = max(RR) - rr_min;
  real rr_sd = sd(RR);
  real t_min = min(t);
  real t_range = max(t) - t_min;

  // Estimate sampling rate from time vector
  real avg_interval = (max(t) - min(t)) / (N - 1);
  real sample_rate = 1.0 / avg_interval; // Samples per minute
}

parameters {
  // --- Double-logistic parameters (unchanged) ---
  real tau_logit;
  real delta_logit;
  real lambda_log;
  real phi_log;

  // --- Baseline and SDNN parameters (unchanged) ---
  real alpha_r_logit;
  real beta_r_logit;
  real c_r_logit;
  real alpha_s_logit;
  real beta_s_logit;
  real c_s_logit;

  // --- Spectral proportion parameters (unchanged) ---
  vector[2] y_base_log;
  vector[2] y_pert_log;
  real c_c_logit;

  // --- Band-limited noise sources ---
  // Instead of frequency amplitudes, we generate band-limited signals directly
  vector[N] z_vlf;  // White noise for VLF band
  vector[N] z_lf;   // White noise for LF band
  vector[N] z_hf;   // White noise for HF band

  // --- Time-varying band gains ---
  vector[N] log_gain_vlf_raw;  // Unconstrained VLF gain
  vector[N] log_gain_lf_raw;   // Unconstrained LF gain
  vector[N] log_gain_hf_raw;   // Unconstrained HF gain

  // --- Smoothness parameters for gains ---
  real<lower=0> sigma_gain;

  // --- Variance split ---
  real w_logit;
}

transformed parameters {
  vector[N] mu;
  vector[N] var_resid;

  // --- Map parameters to natural scale (unchanged) ---
  real tau = inv_logit(tau_logit) * t_range + t_min;
  real delta = inv_logit(delta_logit) * (t_range - tau);
  real lambda = exp(lambda_log);
  real phi = exp(phi_log);

  real alpha_r = inv_logit(alpha_r_logit) * 2 * rr_range + rr_min;
  real beta_r = inv_logit(beta_r_logit) * alpha_r;
  real c_r = inv_logit(c_r_logit) * 2;

  real alpha_s = inv_logit(alpha_s_logit) * rr_sd;
  real beta_s = inv_logit(beta_s_logit) * alpha_s;
  real c_s = inv_logit(c_s_logit) * 2;

  real c_c = inv_logit(c_c_logit);
  real w = inv_logit(w_logit);

  // --- Double-logistic curves (unchanged) ---
  vector[N] D1 = logistic_curve(t, tau, lambda);
  vector[N] D2 = logistic_curve(t, tau + delta, phi);

  // --- Baseline and SDNN trajectories (unchanged) ---
  vector[N] RR_baseline = alpha_r - beta_r .* D1 + (c_r * beta_r) .* D2;
  vector[N] SDNN_t = alpha_s - beta_s .* D1 + (c_s * beta_s) .* D2;

  // --- Spectral proportions (unchanged) ---
  vector[N] C_t = D1 .* (1.0 - c_c .* D2);
  vector[3] pi_base = softmax(append_row(y_base_log, 0.0));
  vector[3] pi_pert = softmax(append_row(y_pert_log, 0.0));
  matrix[N, 3] p_t = (1.0 - C_t) * pi_base' + C_t * pi_pert';

  // --- Generate smooth time-varying gains ---
  vector[N] log_gain_vlf;
  vector[N] log_gain_lf;
  vector[N] log_gain_hf;

  // Random walk prior for smooth evolution
  log_gain_vlf[1] = log_gain_vlf_raw[1];
  log_gain_lf[1] = log_gain_lf_raw[1];
  log_gain_hf[1] = log_gain_hf_raw[1];

  for (n in 2:N) {
    log_gain_vlf[n] = log_gain_vlf[n-1] + sigma_gain * log_gain_vlf_raw[n];
    log_gain_lf[n] = log_gain_lf[n-1] + sigma_gain * log_gain_lf_raw[n];
    log_gain_hf[n] = log_gain_hf[n-1] + sigma_gain * log_gain_hf_raw[n];
  }

  vector[N] gain_vlf = exp(log_gain_vlf);
  vector[N] gain_lf = exp(log_gain_lf);
  vector[N] gain_hf = exp(log_gain_hf);

  // --- Generate band-limited signals ---
  // Option 1: Filter white noise
  vector[N] vlf_signal = bandpass_filter(z_vlf, vlf_low, vlf_high, sample_rate, filter_order);
  vector[N] lf_signal = bandpass_filter(z_lf, lf_low, lf_high, sample_rate, filter_order);
  vector[N] hf_signal = bandpass_filter(z_hf, hf_low, hf_high, sample_rate, filter_order);

  // Option 2: Direct band-limited generation (uncomment to use)
  // vector[N] vlf_signal = generate_band_limited_noise(N, vlf_low, vlf_high, sample_rate);
  // vector[N] lf_signal = generate_band_limited_noise(N, lf_low, lf_high, sample_rate);
  // vector[N] hf_signal = generate_band_limited_noise(N, hf_low, hf_high, sample_rate);

  // Normalize each band to have unit variance (approximately)
  vlf_signal = vlf_signal / sd(vlf_signal);
  lf_signal = lf_signal / sd(lf_signal);
  hf_signal = hf_signal / sd(hf_signal);

  // --- Apply time-varying gains and combine ---
  vector[N] structured_signal =
    gain_vlf .* vlf_signal .* p_t[:,1] +
    gain_lf .* lf_signal .* p_t[:,2] +
    gain_hf .* hf_signal .* p_t[:,3];

  // --- Calculate exact variance allocation ---
  // This ensures proper variance decomposition
  vector[N] target_var_structured = square(SDNN_t) .* w;
  vector[N] current_var_structured = square(gain_vlf) .* square(p_t[:,1]) +
                                    square(gain_lf) .* square(p_t[:,2]) +
                                    square(gain_hf) .* square(p_t[:,3]);

  // Rescale to match target variance
  vector[N] scale_factor = sqrt(target_var_structured ./ (current_var_structured + 1e-12));

  // Final structured signal with correct variance
  vector[N] scaled_structured = structured_signal .* scale_factor;

  // --- Final predictions ---
  mu = RR_baseline + scaled_structured;
  var_resid = square(SDNN_t) .* (1.0 - w);
}

model {
  // --- Priors for double-logistic (unchanged) ---
  tau_logit ~ normal(logit((tau_mu - t_min) / t_range), 0.2);
  delta_logit ~ normal(logit(delta_mu / (t_range - tau_mu)), 0.2);
  lambda_log ~ normal(log(lambda_mu), 0.2);
  phi_log ~ normal(log(phi_mu), 0.2);

  // --- Priors for baseline/SDNN (unchanged) ---
  alpha_r_logit ~ normal(0, 2);
  beta_r_logit ~ normal(0, 2);
  c_r_logit ~ normal(0, 2);
  alpha_s_logit ~ normal(0, 2);
  beta_s_logit ~ normal(0, 2);
  c_s_logit ~ normal(0, 2);

  // --- Priors for spectral proportions (unchanged) ---
  y_base_log ~ normal([0, 0]', 2);
  y_pert_log ~ normal([0, 0]', 2);
  c_c_logit ~ normal(1, 2);

  // --- Priors for noise sources ---
  z_vlf ~ std_normal();
  z_lf ~ std_normal();
  z_hf ~ std_normal();

  // --- Priors for gain smoothness ---
  log_gain_vlf_raw ~ std_normal();
  log_gain_lf_raw ~ std_normal();
  log_gain_hf_raw ~ std_normal();
  sigma_gain ~ exponential(10);

  // --- Prior for variance split ---
  w_logit ~ normal(1.5, 1);

  // --- Likelihood ---
  RR ~ normal(mu, sqrt(var_resid));
}
