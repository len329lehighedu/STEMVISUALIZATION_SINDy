# engine/sindy_model.py

import pysindy as ps
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SINDyEngine:
    def __init__(self):
        self.model = None
        self.feature_names = []

    # ------------------------------------------------------------------
    # Helper: build library
    # ------------------------------------------------------------------
    def _build_library(self, lib_type, poly_degree):
        if lib_type == "Polynomial":
            return ps.PolynomialLibrary(degree=int(poly_degree))
        elif lib_type == "Fourier":
            return ps.FourierLibrary(n_frequencies=int(poly_degree))
        elif lib_type == "Combined":
            return (ps.PolynomialLibrary(degree=int(poly_degree))
                    + ps.FourierLibrary(n_frequencies=int(poly_degree)))
        else:
            raise ValueError(
                f"Inappropriate library type: '{lib_type}'. "
                "Choose: Polynomial, Fourier, Combined."
            )

    # ------------------------------------------------------------------
    # Calculate derivatives on continuous timespan
    # ------------------------------------------------------------------
    def compute_derivatives(self, X, t):
        """
        calculate dx/dt on all data using SmoothedFiniteDifference.
        return dX shape (n_samples, n_features).
        """
        diff = ps.SmoothedFiniteDifference()
        dX = diff(X, t=t)
        return dX

    # ------------------------------------------------------------------
    # Fit model with random split on (X, dX) pairs
    # ------------------------------------------------------------------
    def fit_model(self, X, t, poly_degree, threshold, names,
                  lib_type="Polynomial",
                  train_frac=0.6, random_seed=42, split_method="random",):
        """
        Approach:
          1. Calculate dX on all data
          2. Random shuffle (X, dX) pairs
          3. Fit SINDy on train split
          4. Validate on val split (compare dX_pred vs dX_val)

        Returns: (train_idx, val_idx) to plot later
        """
        self.feature_names = names

        # Step 1: Calculate dx/dt
        dX = self.compute_derivatives(X, t)

        # Step 2: random shuffle indices
        n = len(t)
        n_train = int(n * train_frac)
        # for random block split
        n_blocks = 20
        rng = np.random.default_rng(random_seed)

        if split_method == "random sampling":
            # Random point-level split.
            # Shuffle ALL sample indices, then take the first `n_train` as the
            # training set and the rest as validation. This gives good coverage
            # of the entire state space explored by the trajectory, but each
            # validation point may sit only 1-2 time-steps away from a training
            # point (high autocorrelation between neighbors), so the validation
            # score can be slightly optimistic — it is not a fully independent
            # check on its own (see Test tab for genuine out-of-sample validation).
            indices = rng.permutation(n)

            train_idx = np.sort(indices[:n_train])
            val_idx = np.sort(indices[n_train:])
            # np.sort() restores chronological order within each subset.
            # Without this, concatenating shuffled indices would leave t[train_idx]
            # jumping backward and forward in time — breaking any code downstream
            # that assumes indices are time-ordered (e.g. plotting a line, or
            # forward-simulating with solve_ivp which requires increasing t).

        elif split_method == "time-based":
            # Classic chronological split: first `n_train` points -> train,
            # remaining points -> validation. This mimics a "predict the future
            # from the past" scenario and avoids the autocorrelation leak of
            # random sampling, but at the cost of validation coverage: if the
            # trajectory hasn't fully explored its state space by n_train, the
            # validation set may sit in a region of state space the model never
            # saw during training, which can make validation metrics look worse
            # for reasons unrelated to model quality.
            train_idx = np.arange(n_train)
            val_idx = np.arange(n_train, n)

        elif split_method == "random block":
            # Random BLOCK split — a middle ground between "random sampling"
            # and "time-based".
            #
            # Rationale: random sampling shuffles individual points, so a
            # validation point can sit right next to a training point in time,
            # and because the trajectory changes smoothly, neighboring points
            # are nearly identical (high autocorrelation) — the model effectively
            # "sees" a near-duplicate of each validation point during training,
            # inflating the validation score.
            #
            # Fix: instead of shuffling individual points, chop the trajectory
            # into contiguous blocks (chunks of `block_size` consecutive samples)
            # and shuffle/split at the BLOCK level. As long as block_size is large
            # enough (e.g. >10-20 samples), each validation block sits far enough
            # from its neighbors in time that autocorrelation leak is greatly
            # reduced, while still preserving random coverage across the full
            # trajectory (unlike time-based, which only validates on the tail end).
            # each block's length in samples
            block_size = max(5, n // n_blocks)
            n_blocks_total = n // block_size         # how many whole blocks fit in the data
            # shuffle block IDs, not point indices
            block_ids = rng.permutation(n_blocks_total)
            n_train_blocks = int(n_blocks_total * train_frac)

            train_blocks = sorted(block_ids[:n_train_blocks])
            val_blocks = sorted(block_ids[n_train_blocks:])

            # Expand each block ID into its actual range of point indices, then
            # flatten into a single index array per split.
            train_idx = np.concatenate(
                [np.arange(b*block_size, (b+1)*block_size) for b in train_blocks])
            val_idx = np.concatenate(
                [np.arange(b*block_size, (b+1)*block_size) for b in val_blocks])

            # Same reasoning as the "random sampling" branch above: block IDs were
            # selected in random order, so concatenating them leaves the resulting
            # index array chronologically scrambled BETWEEN blocks (though each
            # individual block is already internally ordered). Sorting restores
            # global time-order so downstream code (plotting, solve_ivp, etc.)
            # can safely assume increasing time — the result has "gaps" where
            # blocks belonging to the other split were skipped, which is expected
            # and fine; only the ordering within the returned array matters.
            train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

        else:
            raise ValueError(
                f"Unknown split_method '{split_method}'. "
                "Use 'random' or 'time'."
            )

        X_train = X[train_idx]
        dX_train = dX[train_idx]
        X_val = X[val_idx]
        dX_val = dX[val_idx]

        # Step 3: fit SINDy with (X_train, dX_train)
        library = self._build_library(lib_type, poly_degree)
        optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=ps.FiniteDifference()  # dummy
        )

        # Fit directly with already calculated x_dot
        self.model.fit(X_train, t=t[train_idx],
                       x_dot=dX_train, feature_names=names)

        # Step 4: calculate metrics on derivative space
        dX_train_pred = self.model.predict(X_train)
        dX_val_pred = self.model.predict(X_val)

        metrics_train = self._metrics_on_dx(dX_train, dX_train_pred)
        metrics_val = self._metrics_on_dx(dX_val,   dX_val_pred)

        return self.model, train_idx, val_idx, metrics_train, metrics_val

    # ------------------------------------------------------------------
    # Metrics on dx/dt space
    # ------------------------------------------------------------------
    def _metrics_on_dx(self, dX_true, dX_pred):
        mse = mean_squared_error(dX_true, dX_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(dX_true, dX_pred))
        r2 = float(r2_score(dX_true, dX_pred, multioutput='uniform_average'))
        residual = dX_true - dX_pred
        rss = np.sum(residual**2)
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2, 'rss': rss}

    # ------------------------------------------------------------------
    # Equations, simulate, metrics on x(t)
    # ------------------------------------------------------------------
    def get_equations(self, precision=3):
        if self.model is None:
            print("⚠ Warning: Model is not fitted.")
            return []

        # 1. Get the list of variable names when fit (names)
        # If not have variable names -> use x1,x2,x3,...
        names = self.feature_names if self.feature_names else [
            f"x{i}" for i in range(len(self.model.equations()))]

        # 2. get the rhs of the equation from pySINDy
        rhs_list = self.model.equations(precision=precision)

        # 3. Construct the equation with the lhs variable name
        full_equations = []
        for i, rhs in enumerate(rhs_list):
            # derivative form: d(x1)/dt
            lhs = f"d({names[i]})/dt"

            # Append: d(x1)/dt = ...
            full_equations.append(f"{lhs} = {rhs}")

        return full_equations

    def simulate(self, x0, t_range):
        if self.model is None:
            print("⚠ Warning: Model is not fitted.")
            return None
        try:
            result = self.model.simulate(x0, t_range)
        except Exception as e:
            raise RuntimeError(f"Simulate failed: {e}")
        if np.any(np.isinf(result)) or np.any(np.isnan(result)):
            raise RuntimeError(
                "Simulation diverged (overflow/nan). "
                "Try increase Sparsity Threshold or decrease Degree."
            )
        return result

    def simulate_with_model(self, model_instance, x0, t):
        try:
            return model_instance.simulate(x0, t)
        except Exception as e:
            print(f"Error when simulate using old model: {e}")
            return np.zeros((len(t), len(x0)))

    def calculate_metrics(self, X_true, X_pred):
        """Metrics on x(t) — use for Test tab."""
        if np.any(np.isinf(X_pred)) or np.any(np.isnan(X_pred)):
            raise ValueError("X_pred contains inf or nan.")
        mse = mean_squared_error(X_true, X_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(X_true, X_pred))
        r2 = float(r2_score(X_true, X_pred, multioutput='uniform_average'))
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2}

    def compute_diagnostics(self, X, t):
        """
        Compute raw diagnostic data for residual analysis after a SINDy fit.
        Returns data arrays for 3 plots — no classification, no thresholds.
        The researcher reads the plots and draws their own conclusions.

        Returns a dict with:
            't'           : time array (shared x-axis for Plot 1)
            'residuals'   : dict {var_name: residual array}  → Plot 1 (Residual vs Time)
            'fft_freqs'   : frequency array (shared x-axis for Plot 2)
            'fft_amps'    : dict {var_name: FFT amplitude}   → Plot 2 (Residual FFT)
            'dX_true'     : dict {var_name: true derivative} → Plot 3 (dX_true vs dX_pred)
            'dX_pred'     : dict {var_name: pred derivative} → Plot 3
            'stats'       : dict {var_name: {snr_db, autocorr, r2_dx}} → shown as text
        """
        if self.model is None:
            return None

        dX_true = self.compute_derivatives(X, t)
        dX_pred = self.model.predict(X)
        residual = dX_true - dX_pred  # shape: (n_samples, n_features)

        n = len(t)
        dt = float(np.mean(np.diff(t)))

        # FFT frequency axis — same for all variables
        fft_freqs = np.fft.rfftfreq(n, d=dt)

        result = {
            't':         t,
            'residuals': {},
            'fft_freqs': fft_freqs,
            'fft_amps':  {},
            'dX_true':   {},
            'dX_pred':   {},
            'stats':     {},
        }

        for i in range(residual.shape[1]):
            r = residual[:, i]
            name = self.feature_names[i] if self.feature_names else f"x{i}"

            # --- Plot 1: Residual vs Time ---
            result['residuals'][name] = r

            # --- Plot 2: FFT of residual ---
            # Amplitude spectrum — dominant frequency reveals missing periodic terms
            fft_amp = np.abs(np.fft.rfft(r)) / n
            result['fft_amps'][name] = fft_amp

            # --- Plot 3: dX_true vs dX_pred scatter ---
            # Perfect model → all points on y=x diagonal
            result['dX_true'][name] = dX_true[:, i]
            result['dX_pred'][name] = dX_pred[:, i]

            # --- Stats shown as text (no classification) ---
            signal_power = np.var(dX_true[:, i])
            noise_power = np.var(r)
            r2_dx = float(
                r2_score(dX_true[:, i], dX_pred[:, i])) if signal_power > 0 else 0.0
            snr_db = 10 * np.log10(signal_power /
                                   noise_power) if noise_power > 0 else 99.0
            r_norm = r - r.mean()
            autocorr = float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1])

            result['stats'][name] = {
                'r2_dx':    round(r2_dx,   3),
                'snr_db':   round(snr_db,  2),
                'autocorr': round(autocorr, 3),
            }

        return result

    # ------------------------------------------------------------------
    # Block bootstrap helper — shared logic between split_method="random
    # block" (fit_model) and the ensemble bootstrap below. Chopping into
    # contiguous blocks (instead of resampling individual points) preserves
    # the local time-autocorrelation structure of the trajectory, which a
    # naive point-level bootstrap would destroy (see project notes on
    # Random Sampling split's autocorrelation leak).
    # ------------------------------------------------------------------

    def _make_blocks(self, n, n_blocks=20):
        block_size = max(5, n // n_blocks)
        n_blocks_total = n // block_size
        return [np.arange(b * block_size, (b + 1) * block_size)
                for b in range(n_blocks_total)]

    # ------------------------------------------------------------------
    # Block-bootstrap ensemble fit — runs STLSQ n_bootstrap times on
    # resampled-with-replacement BLOCKS of the trajectory, to estimate:
    #   - inclusion_pct : how often each candidate term in Θ(X) survives
    #                     thresholding (0 = never, 1 = always)
    #   - coef_mean/std : mean & std of the coefficient, computed ONLY
    #                     over the bootstrap runs where the term was
    #                     non-zero (a term dropped to 0 is a "the model
    #                     chose not to use this" decision, not a small
    #                     coefficient estimate — mixing the two would
    #                     bias the mean toward zero without meaning)
    #
    # Experimental / localhost-only: runs synchronously, no threading.
    # ------------------------------------------------------------------
    def fit_ensemble(self, X, t, poly_degree, threshold, names,
                     lib_type="Polynomial", n_bootstrap=50,
                     random_seed=42, min_inclusion_pct=0.2,
                     progress_callback=None):
        """
        Returns
        -------
        dict {
            'feature_names': [...],   # all candidate term names from Θ(X)
            'per_state': {
                state_name: {
                    'inclusion_pct': {term: float},
                    'coef_mean':     {term: float or None},
                    'coef_std':      {term: float or None},
                    'n_samples':     {term: int},
                }
            },
            'n_bootstrap': n_bootstrap,
        }
        """
        # --- STEP 0: Prepare the full derivative array once ---
        # dX has shape (n_samples, n_states). This is computed ONCE on the
        # full, original trajectory (not per-bootstrap) because the smoothed
        # finite-difference derivative needs a continuous, ordered time axis
        # to be accurate. We will later index into this same dX array using
        # bootstrap-resampled block indices.
        dX = self.compute_derivatives(X, t)
        n = len(t) # total number of time samples in the trajectory
        blocks = self._make_blocks(n) # chop the trajectory into ~20 contiguous chunks 
                                    #(list of index arrays, e.g. [0..49], [50..99], ...)
        n_blocks = len(blocks) # how many blocks we actually got (may be < 20
                                # if n is small, due to the max(5, n//n_blocks) floor)
        rng = np.random.default_rng(random_seed) # dedicated RNG so results are reproducible for a given random_seed

        # --- STEP 1: "Probe" fit — NOT a real result, just to get term names ---
        # We fit once with threshold=0.0 (i.e. no sparsification at all) purely
        # to ask pySINDy: "given this library type + degree, what is the full,
        # fixed list of candidate term names?" (e.g. ['1', 'x', 'y', 'x^2', 'xy', ...]).
        # This list must be IDENTICAL across all bootstrap runs so that we can
        # align/aggregate inclusion counts and coefficients term-by-term.
        # We never look at this probe model's coefficients — only its term names.
        probe_model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.0),
            feature_library=self._build_library(lib_type, poly_degree),
        )
        probe_model.fit(X, t=t, x_dot=dX, feature_names=names)
        term_names = probe_model.get_feature_names() # e.g. ['1', 'x0', 'x1', 'x0^2', 'x0 x1', ...]

        # --- STEP 2: Allocate accumulators ---
        n_states = X.shape[1] # number of state variables (columns of X), e.g. x, y, z
        n_terms = len(term_names) # number of candidate library terms per equation
        
        # inclusion_count[s, k] = how many bootstrap runs kept term k
        # (non-zero coefficient) in the equation for state s.
        inclusion_count = np.zeros((n_states, n_terms))
        
        # coef_records[s][k] = a running Python list of the actual coefficient
        # values observed for term k in state s's equation, across bootstrap
        # runs — but ONLY collected when that term was non-zero in that run
        # (a zero/dropped term is treated as "excluded", not "coefficient ~0").
        coef_records = [[[] for _ in range(n_terms)] for _ in range(n_states)]

        # --- STEP 3: Bootstrap loop — repeat n_bootstrap times ---
        for b in range(n_bootstrap):
            # Resample BLOCKS with replacement: draw n_blocks block-indices,
            # allowing repeats and allowing some blocks to be skipped entirely.
            # This is the "block bootstrap" resampling scheme (as opposed to
            # resampling individual time points, which would break temporal
            # autocorrelation structure within a block)
            chosen_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
            
            # Turn the list of chosen block IDs into an actual flat array of
            # point indices, e.g. block 3 -> [150..199], block 3 again -> [150..199]
            # again (duplicated), block 7 -> [350..399], etc. Order here does
            # NOT need to be globally chronological because we're about to
            # treat this as an independent synthetic dataset with its own
            # dummy time axis (see below) — the derivative was already computed
            # correctly beforehand using the REAL t.
            idx = np.concatenate([blocks[bi] for bi in chosen_blocks])
            
            # Slice out the resampled rows from the ORIGINAL X and dX arrays.
            # Since dX was computed once on the true, continuous trajectory,
            # this reuses accurate derivative estimates instead of recomputing
            # (and potentially corrupting) derivatives on a resampled/duplicated
            # time series.
            X_b, dX_b = X[idx], dX[idx]
            
            # Dummy/fake time axis, just 0, 1, 2, ... matching len(idx).
            # This is needed because ps.SINDy.fit() expects a `t` argument,
            # but since we're passing x_dot=dX_b directly (already computed),
            # this `t` is NOT used to differentiate anything — it's just a
            # placeholder to satisfy the API. Using the real t[idx] here would
            # actually be wrong/misleading, since idx is not monotonic/unique
            # after block-resampling (duplicated blocks -> duplicated timestamps),
            # which could break internal assumptions in pySINDy.
            t_dummy = np.arange(len(idx), dtype=float)

            # Build a FRESH SINDy model for this bootstrap iteration, with the
            # same library type/degree and same sparsity threshold as the
            # "real" fit — we want to know how much the discovered terms
            # fluctuate under IDENTICAL hyperparameters, only the data changed.
            model_b = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold),
                feature_library=self._build_library(lib_type, poly_degree),
                differentiation_method=ps.FiniteDifference()
            )
            try:
                # Fit directly on the resampled block-bootstrap data, again
                # passing x_dot=dX_b explicitly so pySINDy skips its own
                # differentiation step and just does library-fit + STLSQ.
                model_b.fit(X_b, t=t_dummy, x_dot=dX_b, feature_names=names)
            except Exception as e:
                # Some resamples can be degenerate (e.g. STLSQ fails to
                # converge, or a singular matrix from too little variation in
                # the resampled data). Rather than crashing the whole ensemble,
                # log it and simply skip this bootstrap iteration — it will
                # not contribute to inclusion_count or coef_records, and
                # n_bootstrap (used as the denominator later) stays the
                # originally requested count.
                print(f"[Ensemble] bootstrap {b} failed: {e}")
                continue

            # coefs is a (n_states, n_terms) matrix: coefs[s, k] is the
            # coefficient this bootstrap run assigned to term k in the
            # equation for state s (0 if STLSQ thresholded it away).
            coefs = model_b.coefficients()  # shape (n_states, n_terms)
            for s in range(n_states):
                for k in range(n_terms):
                    if coefs[s, k] != 0:
                        # Term k survived thresholding in this run for state s:
                        # count it as "included" and record its coefficient value.
                        inclusion_count[s, k] += 1
                        coef_records[s][k].append(coefs[s, k])

            if progress_callback:
                progress_callback(b + 1, n_bootstrap)

        # --- STEP 4: Aggregate results into the final per-state dict ---
        result = {'feature_names': term_names,
                  'per_state': {}, 'n_bootstrap': n_bootstrap}
        
        for s in range(n_states):
            # Resolve a readable name for this state (fallback to x0, x1, ...if no names were provided).
            state_name = names[s] if names else f"x{s}"
            # inclusion percentage - coefficient mean - coefficient standard deviation - n samples
            incl_pct, coef_mean, coef_std, n_samp = {}, {}, {}, {}
            
            for k, term in enumerate(term_names):
                # Fraction of bootstrap runs (out of the REQUESTED n_bootstrap,
                # not just the successful ones) in which this term was non-zero
                # for this state. E.g. 0.84 means "included in 84% of runs"
                incl_pct[term] = float(inclusion_count[s, k] / n_bootstrap)
                vals = coef_records[s][k] # list of collected coefficient values
                n_samp[term] = len(vals) # how many runs actually produced a value
                
                # Only report a mean/std coefficient if the term was "reliably"
                # present (inclusion frequency >= min_inclusion_pct) AND we
                # actually have at least one recorded value. This avoids
                # reporting a misleading mean/std computed from just 1-2
                # occurrences out of 50 runs.
                if incl_pct[term] >= min_inclusion_pct and vals:
                    coef_mean[term] = float(np.mean(vals))
                    coef_std[term] = float(np.std(vals))
                else:
                    # Term is too rare/unstable to summarize meaningfully —
                    # report None instead of a noisy or empty-array statistic.
                    coef_mean[term] = None
                    coef_std[term] = None
            # Store all four dicts for this state under its name.
            result['per_state'][state_name] = {
                'inclusion_pct': incl_pct, 'coef_mean': coef_mean,
                'coef_std': coef_std, 'n_samples': n_samp,
            }
        return result
