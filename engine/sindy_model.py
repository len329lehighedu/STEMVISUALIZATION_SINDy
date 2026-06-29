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

        dX_true  = self.compute_derivatives(X, t)
        dX_pred  = self.model.predict(X)
        residual = dX_true - dX_pred  # shape: (n_samples, n_features)

        n  = len(t)
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
            r    = residual[:, i]
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
            noise_power  = np.var(r)
            r2_dx   = float(1 - noise_power / signal_power) if signal_power > 0 else 0.0
            snr_db  = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 99.0
            r_norm  = r - r.mean()
            autocorr = float(np.corrcoef(r_norm[:-1], r_norm[1:])[0, 1])

            result['stats'][name] = {
                'r2_dx':    round(r2_dx,   3),
                'snr_db':   round(snr_db,  2),
                'autocorr': round(autocorr, 3),
            }

        return result