# engine/sindy_model.py

import pysindy as ps
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SINDyEngine:
    def __init__(self):
        self.model         = None
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
            return (ps.PolynomialLibrary(degree=2)
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
        dX   = diff._differentiate(X, t)
        return dX

    # ------------------------------------------------------------------
    # Fit model with random split on (X, dX) pairs 
    # ------------------------------------------------------------------
    def fit_model_random_split(self, X, t, poly_degree, threshold, names,
                               lib_type="Polynomial",
                               train_frac=0.6, random_seed=42):
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
        rng     = np.random.default_rng(random_seed)
        indices = rng.permutation(n)

        n_train    = int(n * train_frac)
        train_idx  = np.sort(indices[:n_train])   # sort để giữ thứ tự time
        val_idx    = np.sort(indices[n_train:])

        X_train  = X[train_idx]
        dX_train = dX[train_idx]
        X_val    = X[val_idx]
        dX_val   = dX[val_idx]

        # Step 3: fit SINDy with (X_train, dX_train)
        library   = self._build_library(lib_type, poly_degree)
        optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library,
            differentiation_method=ps.FiniteDifference()  # dummy
        )

        # Fit directly with already calculated x_dot
        self.model.fit(X_train, t=t[train_idx], x_dot=dX_train, feature_names=names)

        # Bước 4: calculate metrics on derivative space
        dX_train_pred = self.model.predict(X_train)
        dX_val_pred   = self.model.predict(X_val)

        metrics_train = self._metrics_on_dx(dX_train, dX_train_pred)
        metrics_val   = self._metrics_on_dx(dX_val,   dX_val_pred)

        return self.model, train_idx, val_idx, metrics_train, metrics_val

    # ------------------------------------------------------------------
    # Metrics on dx/dt space
    # ------------------------------------------------------------------
    def _metrics_on_dx(self, dX_true, dX_pred):
        mse  = mean_squared_error(dX_true, dX_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(dX_true, dX_pred))
        r2   = float(r2_score(dX_true, dX_pred, multioutput='uniform_average'))
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2}

    # ------------------------------------------------------------------
    # Equations, simulate, metrics on x(t)
    # ------------------------------------------------------------------
    def get_equations(self, precision=3):
        if self.model is None:
            print("⚠ Warning: Model is not fitted.")
            return []
        
        # 1. Get the list of variable names when fit (names)
        # If not have variable names -> use x1,x2,x3,...
        names = self.feature_names if self.feature_names else [f"x{i}" for i in range(len(self.model.equations()))]
        
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
        mse  = mean_squared_error(X_true, X_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(X_true, X_pred))
        r2   = float(r2_score(X_true, X_pred, multioutput='uniform_average'))
        return {'mse': float(mse), 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    def _estimate_lyapunov(self, X, t):
        """
        Estimate the largest Lyapunov exponent (λ) from observed trajectory data.

        Core idea:
            In a chaotic system, two initially close trajectories diverge exponentially.
            λ measures the average rate of that divergence:
                λ = mean( log(d1 / d0) / dt )
            where d0 is the initial distance between two neighbors,
            and d1 is their distance one timestep later.

            λ > 0  → trajectories diverge → system is chaotic
            λ ≤ 0  → trajectories converge or stay close → system is stable

        Algorithm (Kantz / Rosenstein simplified):
            1. Sample ~50 reference points along the trajectory
            2. For each reference point i, find its nearest neighbor j
            (excluding temporal neighbors within a window to avoid trivial pairs)
            3. Compute d0 = ||X[i] - X[j]||  (initial separation)
            4. Compute d1 = ||X[i+1] - X[j+1]||  (separation one step later)
            5. Local exponent = log(d1 / d0) / dt
            6. λ = mean of all local exponents

        Limitations:
            - Requires sufficiently long, low-noise trajectory
            - Short or noisy data will produce unreliable estimates
            - This is a rough estimate, not a rigorous Lyapunov calculation
        """
        n  = len(t)
        dt = float(np.mean(np.diff(t)))
        exponents = []

        for i in range(0, n - 1, max(1, n // 50)):  # sample ~50 reference points
            # Find nearest neighbor in state space (not in time)
            dists = np.linalg.norm(X - X[i], axis=1)

            # Exclude temporal neighbors to avoid trivially close pairs
            dists[max(0, i - 5): i + 5] = np.inf

            j  = np.argmin(dists)
            d0 = dists[j]
            if d0 < 1e-10:
                continue  # skip degenerate pairs (identical points)

            # Measure divergence one timestep later
            if i + 1 < n and j + 1 < n:
                d1 = np.linalg.norm(X[i + 1] - X[j + 1])
                if d1 > 0 and d0 > 0:
                    exponents.append(np.log(d1 / d0) / dt)

        return float(np.mean(exponents)) if exponents else 0.0
    
    def analyze_residual(self, X, t):
        """
        Diagnose SINDy fit quality by analyzing the residual:
            residual = dX_true - dX_predicted

        If the model were perfect, residual would be pure white noise.
        Any structure remaining in the residual means the model missed something.

        Three diagnostic signals are computed per state variable:

        1. R² in derivative space (r2_dx)
        Measures how much of the derivative variance the model explains.
        Used as the primary gate: if r2_dx is high, the model fit is good
        in derivative space — failure is then about simulation stability,
        not about the library or data quality.

        2. SNR (Signal-to-Noise Ratio, in dB)
        Measures how much of the derivative energy the model explains
        relative to what it failed to explain.
            signal_power = var(dX_true)
            noise_power  = var(residual)
            SNR = 10 * log10(signal_power / noise_power)
        Low SNR → residual energy is close to signal energy → data too noisy.

        3. Lag-1 Autocorrelation of residual
        Pure noise has autocorr ≈ 0 (each timestep is independent).
        A large autocorr means the residual has a repeating pattern →
        a term exists in the true dynamics that the library never offered.

        Failure classification (priority order):
            R² ≥ 0.85 + λ > 0.05  →  CHAOTIC_SYSTEM
                Model found good equations, but the system is inherently chaotic.
                Trajectory divergence is mathematical, not a model error.

            R² ≥ 0.85 + λ ≤ 0.05  →  OK
                Model fits well and system is stable.

            SNR < 10 dB            →  DATA_QUALITY
                Data is too noisy to draw further conclusions.

            autocorr > 0.4         →  LIBRARY_TOO_SIMPLE
                Residual still has structure → library is missing terms.

            else                   →  UNDERFITTING
                R² is low but no clear pattern in residual →
                sparsity threshold is likely too aggressive.
        """
        if self.model is None:
            return None

        dX_true  = self.compute_derivatives(X, t)
        dX_pred  = self.model.predict(X)
        residual = dX_true - dX_pred  # shape: (n_samples, n_features)

        # Estimate Lyapunov exponent once for the full trajectory
        # (system-level property, not per-variable)
        lyap = self._estimate_lyapunov(X, t)

        results = {}
        for i in range(residual.shape[1]):
            r    = residual[:, i]
            name = self.feature_names[i] if self.feature_names else f"x{i}"

            # --- Signal 1: R² in derivative space ---
            # Fraction of derivative variance explained by the model.
            # Primary gate for the classifier — checked before anything else.
            signal_power = np.var(dX_true[:, i])
            noise_power  = np.var(r)
            r2_dx = float(1 - noise_power / signal_power) if signal_power > 0 else 0.0

            # --- Signal 2: SNR (dB) ---
            # How large is the unexplained residual relative to the true signal?
            # Low SNR → data itself is too noisy, not a library problem.
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 99.0

            # --- Signal 3: Lag-1 autocorrelation of residual ---
            # Subtract mean so we measure correlation of fluctuations, not offset.
            # High autocorr → residual has temporal structure → missing library terms.
            r_norm   = r - r.mean()
            autocorr = float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1])

            # --- Priority-based classifier ---
            if r2_dx >= 0.85:
                # Model explains derivative space well.
                # Only remaining question: is the system chaotic?
                if lyap > 0.05:
                    failure = "CHAOTIC_SYSTEM"
                else:
                    failure = "OK"

            elif snr_db < 10:
                # Residual energy is close to signal energy.
                # Data is too noisy — no further diagnosis is reliable.
                failure = "DATA_QUALITY"

            elif abs(autocorr) > 0.4:
                # Residual still has repeating structure despite low R².
                # The library is missing terms that could explain this pattern.
                failure = "LIBRARY_TOO_SIMPLE"

            else:
                # R² is low but residual looks like noise — no clear structure.
                # Most likely cause: sparsity threshold pruned valid terms.
                failure = "UNDERFITTING"

            results[name] = {
                'r2_dx':   round(r2_dx,   3),
                'snr_db':  round(snr_db,  2),
                'autocorr': round(autocorr, 3),
                'lyap':    round(lyap,    4),
                'failure': failure,
            }

        return results