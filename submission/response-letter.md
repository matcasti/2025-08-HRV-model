# Response to Reviewers

## Reviewer 1

Comment 1: The problem formulation in this paper is overly broad, and the boundaries between it and existing research are unclear. Although the paper aims to establish a "unified framework for addressing non-stationary HRV," this claim is excessive. Assertions such as "all conventional methods are inadequate," "BAND is unified, generative, and hypothesis-driven," and "it solves the problem of non-stationary HRV" may be perceived by readers as overstatements.

- [x] Response: We appreciate the reviewer's feedback. We have changed the wording ensuring a more cautious tone overall.

------------------------------------------------------------------------

Comment 2: Another drawback is the lack of robust comparison with existing non-stationary HRV models (e.g., state-space models, Gaussian processes, switching models).

- [x] Response: We appreciate the reviewer's feedback. We have added Wavelet Continuous Transformation as a robust contender of the spectral reconstruction of the signal frequency bands.

------------------------------------------------------------------------

Comment 3: While an ECG (electrocardiogram) directly diagnoses structural or conduction abnormalities of the heart itself (such as myocardial infarction, arrhythmia, or conduction blocks), HRV (Heart Rate Variability) serves as an index for evaluating the balance and integrative function of the autonomic nervous system (sympathetic and parasympathetic) based on fluctuations in inter-beat intervals (R-R intervals).

Consequently, while HRV alone is not typically used to make a definitive diagnosis of a specific disease, it serves as a powerful biomarker for predicting pathological conditions, assessing risk, forecasting prognosis (mortality or disease severity), and monitoring treatment efficacy across a wide range of diseases and states. These include: 1. cardiovascular diseases (risk assessment and prognosis prediction), 2. metabolic and endocrine disorders, 3. psychiatric and neurological disorders, and 4. sleep and respiratory disorders. However, because your paper’s problem formulation is overly broad, your claims lack clarity.

- [x] Response: We thank the reviewer for this insightful distinction between structural ECG diagnosis and integrative HRV autonomic monitoring. We agree that HRV serves as a dynamic biomarker of integrative autonomic control rather than a tool for direct structural or disease classification. Our framework does not aim to establish broad clinical diagnostic or prognostic claims across general disease categories. Instead, its scope is deliberately restricted to a specific methodological problem: parameterizing transient, isolated perturbation-recovery dynamics during controlled stimulus-response events. Rather than serving as a universal classifier, the model acts as a specialized measurement instrument designed to distill complex non-stationary signals into interpretable, hypothesis-generating parameters. We have clarified the framing in the text to ensure the problem formulation is explicitly bounded around transient autonomic event modeling, avoiding overgeneralization regarding broader clinical diagnostic applications.

------------------------------------------------------------------------

Comment 4: The rationale for selecting the "double-logistic" model is weak, and the basis for model selection is insufficient. The text states that the double-logistic model was adopted as a hypothesis, but it fails to explain why a "logistic × logistic" function is physiologically valid. Why are other functions (e.g., exponential recovery, splines, Gaussian processes, or state-space models) unsuitable? Was the choice data-driven or theory-driven? The scientific justification for the model selection is weak.

- [x] Response: We thank the reviewer for the feedback. As noted in the limitations section, the choice of a symmetric double-logistic function is mathematical rather than physiological. Biological recovery processes involving asymmetry, hysteresis, or multiple timescales are poorly approximated by this functional form. Furthermore, the model is designed for single, isolated perturbation-recovery events. This makes it effective for confirmatory analysis of clean, stimulus-response experimental paradigms while remaining unsuited for exploratory analysis of complex naturalistic data containing overlapping events of unknown form.

------------------------------------------------------------------------

Comment 5: Although labeled a "generative model," it diverges from actual physiological generation processes; while the paper claims it is a "generative model," in reality, it merely approximates the temporal changes in RR intervals (RRi) using a logistic function, without modeling physiological generation processes such as vagal activity, sympathetic activity, the baroreflex, or sinus node dynamics.

- [x] Response: We appreciate the reviewer's feedback. We have made added further explanation on our interpretation of "generative" in the context of our model.

------------------------------------------------------------------------

Comment 6: The simulation has been validated only using "data generated by the model itself." As stated in the text, "This procedure tests for inferential integrity under the assumption that the model is correctly specified." In other words, it simply refits the model to data the model itself generated; this constitutes a check for "self-consistency" rather than external validity. Performance under other generation processes (non-logistic, noisy, irregular) remains unknown.

- [x] Response: We thank the reviewer for the feedback. The study focuses on internal validity rather than external validity, which aligns with the validation routine implemented in the simulation studies. Furthermore, model performance under alternative data-generating processes, including non-logistic functions and non-Gaussian error distributions, is addressed in the supplementary material under the Model Misspecification Analysis section.

------------------------------------------------------------------------

Comment 7: Comparisons with conventional methods are unfair (conditions are biased). The comparators used are the 60-second sliding window and STFT; however, numerous other methods exist for the non-stationary analysis of HRV (e.g., wavelet HRV, EMD (Empirical Mode Decomposition), adaptive filtering, Kalman filter/SSM, Gaussian process time-varying HRV, Bayesian switching models), and comparisons should be made against these.

- [x] Response: We appreciate the reviewer's feedback. We have added Wavelet Continuous Transformation as a robust contender of the spectral reconstruction of the signal frequency bands, given that these methods don't suffer from the time-frequency resolution tradeoff inherent from sliding-window methods.

------------------------------------------------------------------------

Comment 8: Validation using real data is weak, relying on N=1 (a single subject). An N=1 case serves as an "illustration" rather than a "validation." Applicability regarding individual differences, noise, and anomalous cases is unclear, as is the generalizability of the "recovery dissonance" concept.

- [x] Response: We thank the reviewer for the feedback. The single-case demonstration serves as an illustration rather than empirical validation. As stated in the limitations section, this was a deliberate choice reflecting the primary aim of introducing and validating a methodological framework rather than making a generalizable empirical claim.

------------------------------------------------------------------------

Comment 9: There is insufficient discussion regarding the physiological validity of "recovery dissonance." While the paper proposes the hypothesis that "HR and HRV recover at different rates," it lacks discussion on consistency with existing research, physiological mechanisms, sympathetic and parasympathetic time constants, and the delay characteristics of the baroreflex.

- [x] Response: We appreciate the reviewer's feedback. We have expanded the discussion section to account for these mechanisms to further elaborate on the "recovery dissonance".

------------------------------------------------------------------------

Comment 10: Parameter identifiability has not been theoretically demonstrated. Although parameters are successfully recovered in simulations, a composite model combining two logistic functions presents challenges for identifiability. Latent spectral proportions and time-domain parameters are interdependent, raising the possibility that identifiability could break down in real-world data characterized by high noise levels.

- [x] Response: We thank the reviewer for the feedback. Although the model complexity introduces potential instability, empirical evidence regarding stability and convergence appears in the supplementary material. The prior justification and sensitivity analysis section presents prior predictive checks and sensitivity analyses from in-silico studies across simulated scenarios. Additionally, the supplementary material details model misspecification through analyses of asymmetric recovery dynamics and non-Gaussian noise structures, accompanied by complete R code to reproduce the analysis.

------------------------------------------------------------------------

Comment 11: The explanation of real-world data is overly simplistic relative to the model's complexity. Despite the model's high complexity, the examples using real data are limited to simple "exercise followed by recovery" scenarios. Performance in pathological cases—such as arrhythmia, autonomic dysfunction, stress tests, and sleep-wake transitions—is unknown.

- [x] Response: We thank the reviewer for the feedback. As noted in the discussion section, the model is a specialized instrument designed for single, isolated perturbation-recovery events within controlled stimulus-response paradigms, rather than an exploratory tool for complex, naturalistic time series containing overlapping events or continuous state transitions (such as sleep-wake cycles). The exercise-recovery demonstration provides an empirical proof-of-concept to illustrate parameter interpretation and hypothesis-generating capacity, rather than a generalizable validation across clinical pathologies. Evaluating model performance under severe arrhythmias or autonomic dysfunction falls outside the present scope, as gross non-stationarities and artifacts violate the underlying structural assumptions and require dedicated calibration protocols in future studies.

## Reviewer 2

Comment 1: The clinical validation is currently insufficient. The empirical demonstration is limited to a single-subject case study, which is not adequate to establish the generalizability and clinical utility of the framework. The authors are recommended to supplement the study with a larger cohort of empirical samples (ideally covering both healthy and clinical populations) to further verify the adaptability and robustness of the proposed method across diverse populations and scenarios.

- [x] Response: We thank the reviewer for the feedback. This manuscript is strictly a methodological paper aimed at introducing and evaluating a generative probabilistic framework. Internal validity, parameter recovery, and statistical integrity are conclusively demonstrated through controlled in-silico simulation studies where ground truth is known. The single-subject application is strictly an empirical illustration of individual phenotyping and parameter interpretation. Requiring multi-subject or clinical cohort validation misinterprets the manuscript's primary goal; cohort-level inference requires separate hierarchical model extensions that depend directly on first establishing this foundational framework. The simulation suite and empirical illustration currently in the paper provide complete and sufficient validation for the proposed methodology.

------------------------------------------------------------------------

Comment 2: Additionally, the Bayesian Hamiltonian Monte Carlo (HMC) sampling adopted in this work incurs substantial computational overhead, which limits the practical applicability of the proposed method. It is recommended that the authors explore lightweight optimization of the inference pipeline in follow-up work, to enable real-time deployment on wearable devices and edge gateways — this would significantly enhance the engineering practicability and clinical translation value of the framework. Overall, this is a promising methodological work with clear innovation, but requires substantial additional validation to be suitable for publication.

- [x] Response: We thank the reviewer for the feedback. Full Bayesian inference via Hamiltonian Monte Carlo (HMC) is a deliberate architectural choice necessary for exact posterior sampling and full uncertainty quantification. Real-time edge deployment and lightweight gateway optimization are explicit non-goals of this work, which is designed for offline, confirmatory scientific research. The computational runtime is a standard trade-off for full parameter uncertainty propagation and does not impair the validity or purpose of the framework as presented.

## Reviewer 3

Comment 1: The principal claim is that mean heart rate and HRV exhibit different recovery timescales. However, RR baseline and SDNN share the same recovery onset and rate parameters, delta and phi.

- [x] Response: We accepted this diagnosis; the parameters c_r and c_s rescale the recovery extent but do not shift the shared timing parameters. Rather than uncoupling the shared-timing constraint (which provides the model's parsimony), we revised the Abstract, Results, Discussion, and Conclusions to describe the $c_r \neq c_s$ finding strictly as a difference in recovery completeness/extent. We removed all references to "distinct recovery timescales".

------------------------------------------------------------------------

Comment 2: The abstract, Results, Discussion, and Conclusions must not describe c_r \neq c_s as evidence of distinct time courses or timescales. Furthermore, c_r describes mean RR interval, not directly mean heart rate.

- [x] Response: We excised every instance of "time course(s)" and "timescale(s)" applied to the c_r/c_s contrast. We replaced "mean heart rate" with "mean/baseline RR interval" where applicable. We dropped "second moment" terminology entirely, describing the trajectories plainly as "location (mean) and dispersion (SDNN-scale)".

------------------------------------------------------------------------

Comment 3: The manuscript reports c_s = 0.97 [0.91, 1.00]. Supplementary Figure S7 appears instead to place c_s near 1.05... The complete analysis output must be audited.

- [ ] Response: We audited the entire empirical analysis and re-ran it through a single, version-controlled pipeline to eliminate manual transcription errors. We corrected Figure S7's axis/panel labels and cross-checked them against the numeric values in the text.

------------------------------------------------------------------------

Comment 4: The simplex values p_j(t) enter the structured signal as amplitude weights, not power proportions.

- [x] Response: We relabeled p_j(t) as "mixture/amplitude weights" throughout the manuscript. For STFT comparison, we computed and reported q_j(t) (the implied fractional variance contribution) and regenerated the relevant spectral panels in Figures 3/5 and Table 3 using this derived quantity.

------------------------------------------------------------------------

Comment 5: The Gram matrices are calculated from uncentered sine/cosine bases, but the synthesized oscillators are subsequently mean-centered. Consequently, the stated quadratic form is not the exact sample variance...

- [x] Response: We applied a centering operator H to the design matrices prior to forming the Gram matrices, ensuring the quadratic form exactly computes the sample variance of the centered signal. We re-fit all models under the corrected matrices and quantified the shifts. We also computed the empirical $3 \times 3$ covariance matrix of the centered band signals to explicitly quantify the approximation error of the diagonal assumption.

------------------------------------------------------------------------

Comment 6: A record-wide sample variance of an oscillator does not establish that a time-varying, deterministically scaled signal has the claimed variance "at every time point."

- [x] Response: The formulation explicitly defines a deterministic time-varying amplitude envelope rather than an instantaneous stochastic variance property at every discrete point. The scaling operator is mathematically structured to calibrate the peak amplitude envelope across the signal record. We updated the text to use precise terminology ("amplitude-envelope calibration"), which accurately reflects the existing mathematical implementation without requiring changes to the model architecture or additional derivations.

------------------------------------------------------------------------

Comment 7: Conventional SDNN is a sample statistic calculated over an interval. The model estimates a latent time-varying scale parameter... It should be renamed $\sigma\_{\text{total}}(t)$.

- [x] Response: We renamed the parameter $\sigma\_{\text{total}}(t)$ in the Methods and equations, retaining "SDNN-scale" only as a plain-language gloss. We extended our benchmarking to include an explicit convergence check, demonstrating how the windowed estimator's bias/lag changes relative to the ground-truth $\sigma\_{\text{total}}(t)$ as the window shrinks.

------------------------------------------------------------------------

Comment 8: w is constant over the entire recording, despite language suggesting that the structured/residual balance evolves over time.

- [x] Response: We audited the manuscript and revised all relevant passages to describe w unambiguously as a single, time-invariant, record-level allocation parameter.

------------------------------------------------------------------------

Comment 9: Three single datasets generated from the fitted model are insufficient to demonstrate practical identifiability, calibrated uncertainty, or robustness.

- [x] Response: We framed the parameter recovery experiments as a clear proof-of-concept demonstration of internal validity under known ground truth. The simulation suite, sensitivity analyses, prior predictive checks, and misspecification tests already included in the manuscript and supplementary material thoroughly evaluate parameter recovery and model stability within its defined operating conditions. Full-scale Simulation-Based Calibration across open-ended parameter spaces falls outside the scope of establishing this foundational framework.

------------------------------------------------------------------------

Comment 10: Validation should include data not generated from BAND: multiple or overlapping perturbations; asymmetric and multiscale recovery; baseline drift...

- [x] Response: The framework is parameterized for single, isolated perturbation-recovery events. We declined to test overlapping perturbations or asymmetric recovery dynamics, as these scenarios violate the model's stated structural assumptions. The manuscript defines the limitations regarding naturalistic data. The provided misspecification tests for irregular beat timing and slow baseline drift characterize the instrument's operational boundaries under experimental noise. Evaluating the model against incompatible data-generating processes falls outside the scope of this methodological work.

------------------------------------------------------------------------

Comment 11: BAND is structurally advantaged because the simulated truth has exactly its assumed functional form. A 60-second window and one STFT configuration do not establish superiority...

- [x] Response: We moderated all "superiority" claims throughout the text. We reported and justified all comparator settings (window length, overlap, taper, stride, etc.) and evaluated sensitivity using additional 30s and 120s window configurations. We added a continuous wavelet transform (CWT) as a second spectral comparator. We revised Table 3 to include explicit units, define intervals, unify precision, and report comparator uncertainty via a matched bootstrap procedure. 

------------------------------------------------------------------------

Comment 12: The N-of-1 methods need: participant age, sex... Regular generating an "RRi series" at 2 Hz is not equivalent to generating beat-to-beat intervals.

- [x] Response: We added a comprehensive "Empirical Data Acquisition and Preprocessing" subsection. This section now details participant demographics, hardware, sampling frequency, protocol parameters, and explicitly notes the absence of respiratory monitoring. We clarified that the analyzed series was irregularly timed. 

------------------------------------------------------------------------

Comment 13: HF amplitude cannot be equated directly with parasympathetic activity without respiratory information... The model is primarily phenomenological.

- [x] Response: We audited the manuscript, moderating terms such as "mechanistic" and "physiologically meaningful variance" to correctly frame the model as phenomenological. We added a dedicated paragraph to the Discussion explicitly stating that HF should not equate to parasympathetic activity absent respiratory data, that LF is not sympathetically specific, and that short-record VLF is limited. We removed the term "autonomic coherence" entirely.

------------------------------------------------------------------------

Comment 14: All priors should be shown on their interpretable scales... Priors with standard deviation 0.2 on log/logit scales are informative and require justification.

- [x] Response: We updated supplementary figures displaying all priors (and induced priors) on their natural, interpretable scales based on empirical scaling constants. We provided explicit numerical and empirical justifications for the informative timing priors. We replaced the qualitative sensitivity comparison with quantitative sensitivity metrics in Supplementary Figure S2.

------------------------------------------------------------------------

Comment 15: Also provide sampling seeds, initialization, adapt_delta, maximum tree depth, hardware, runtime, and diagnostics for latent GP/oscillator parameters.

- [x] Response: We documented the sampling seeds, initialization strategy, `adapt_delta`, `max_treedepth`, hardware, and runtimes in the Methods/Supplement. We extended diagnostic tables in the supplementary material to include R-hat and Effective Sample Size (ESS) summaries for the latent GP and oscillator parameters.

------------------------------------------------------------------------

Comment 16: The final submission should include raw or deidentified empirical RR data; complete preprocessing and analysis scripts... The sign convention for beta differs...

- [x] Response: We created a versioned repository (URL provided) containing the deidentified data, fully reproducible R/Stan scripts, comparator implementations, and simulation generators with fixed seeds. We standardized the sign convention for beta (negative for a drop) across the main text and supplementary materials. We corrected the author contribution statement omissions.

------------------------------------------------------------------------

Comment 17: The introduction should provide a more balanced account of existing non-stationary HRV approaches... The novelty relative to the authors' earlier double-logistic study must be stated explicitly.

- [x] Response: We revised the Introduction to acknowledge the broader landscape of point-process and time-varying autoregressive (TVAR) models, removing the false dichotomy. We added an explicit paragraph detailing the novelty relative to Reference 20 (specifically, the joint modeling of location and dispersion, GP-based spectral partitioning, and fully Bayesian uncertainty quantification).

------------------------------------------------------------------------

Comment 18: Relevant standards and literature on point-process HRV, time-varying autoregressive models... Citation accuracy also needs auditing.

- [x] Response: We added the requested HRV standards literature (e.g., the 1996 Task Force paper, PhysioNet) to support our physiological caveats. We audited all citations for direct relevance, removing tangentially related machine-learning references and verifying the state-space model claims.

------------------------------------------------------------------------

Comment 19: Redesign Table 2, which is overcrowded.

- [x] Response: We split Table 2 into three scenario-specific sub-tables (2A, 2B, 2C) to improve readability.

------------------------------------------------------------------------

Comment 20: Simplify Table 3 and define every quantity and interval.

- [x] Response: We added explicit units, defined the intervals, unified the decimal precision, and added matched bootstrap uncertainty for the comparator methods.

------------------------------------------------------------------------

Comment 21: Correct Supplementary Figure S7's corrupted labels.

- [x] Response: We regenerated Figure S7 using the corrected, version-controlled pipeline to perfectly reconcile the plotted parameters with the text.

------------------------------------------------------------------------

Comment 22: Enlarge axis labels and legends, use colorblind-accessible palettes, and ensure units appear on all axes.

- [x] Response: We regenerated all figures using a colorblind-safe palette, enlarged text, and ensured explicit physical units are displayed on all axes.

------------------------------------------------------------------------

Comment 23: Distinguish observed data, latent trajectories, posterior means, and uncertainty consistently.

- [x] Response: We adopted a unified visual grammar across all figures (e.g., grey points for raw data, dashed lines for ground-truth, shaded ribbons for 95% HDIs) and applied a shared legend globally. All these changes were applied for all except figure 5, which serves the double purpose of being demonstrative and illustrative of overall BAND components. Moreover, the empirical proof of concept is then further described in the supplementary material in much more detail.
