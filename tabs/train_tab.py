# =============================================================================
# tabs/train_tab.py
#
# PURPOSE
# -------
# Renders the "Train & Validate" tab of the SINDy Expert System.
# Responsibilities:
#   1. Let the user pick a dataset (pre-set system or custom CSV upload).
#   2. Run an automatic data-analysis heuristic ("AI Suggester") that
#      recommends starting values for polynomial degree and sparsity
#      threshold based on linearity / periodicity / noise level.
#   3. Fit a SINDy model on a random train/validation split.
#   4. Show the fitted trajectory, a leaderboard of all past runs, and
#      residual diagnostics (time-domain, frequency-domain, and
#      true-vs-predicted scatter) so the user can judge model quality.
#   5. Allow viewing/deleting any past run from the leaderboard.
# =============================================================================

from bokeh.models import (ColumnDataSource, Slider, Div, Button,
                          Select, DataTable, TableColumn, HTMLTemplateFormatter, FileInput, TextInput, CheckboxButtonGroup, HoverTool)
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import os
import copy
import base64
import io
import warnings
from engine.suggester import analyze_data_linearity


def train_tab_layout(engine, trained_model_storage):
    """
    Build the Bokeh layout for the Train & Validate tab.

    Parameters
    ----------
    engine : SINDyEngine
        Shared engine instance (holds the pySINDy model, fit/simulate/
        diagnostics methods). Same instance is passed to Test/Predict tabs
        so that trained models can be reused across tabs.
    trained_model_storage : dict
        Shared in-memory store: {run_id: {model_instance, metrics, plot_data,
        diagnostics, ...}}. Acts as the "database" for the leaderboard and
        is read by the Test/Predict tabs to let the user pick a trained run.

    Returns
    -------
    bokeh.layouts.column
        The complete tab layout, ready to be added to a Bokeh document.
    """

    def apply_suggestion(df, prefix_msg=""):
        """
        Run analyze_data_linearity() and push the result into the UI:
        updates poly_s / thr_s slider values and displays the reasoning
        text in upload_status. (library is intentionally NOT auto-applied,
        see note in Step 6 above.)
        """
        lib, deg, thr, reason = analyze_data_linearity(df)
        poly_s.value = deg
        thr_s.value = thr
        upload_status.text = f"{prefix_msg}<br><b>Suggestion:</b> {reason}"

    # =========================================================================
    # SECTION 1 — DATA SOURCE SELECTION
    # Dropdown for pre-set systems + custom CSV upload widget.
    # 2 pre-set systems for users to test SINDy's ability
    # =========================================================================

    system_options = [
        ("cs_train_data.csv",   "Coupled Spring-Mass (Pre-set)"),
        ("vanderpol_train.csv", "Van der Pol Oscillator (Pre-set)"),
        ("custom_upload",       "Upload your own data")
    ]

    file_select = Select(title="SELECT SYSTEM", options=system_options,
                         value="cs_train_data.csv")
    # value -> friendly label lookup, used to populate the "Data File"
    # column in the leaderboard (e.g. "cs_train_data.csv" -> "Coupled
    # Spring-Mass (Pre-set)").
    _SYSTEM_LABELS = dict(system_options)

    # File upload widget — hidden until the user picks "Upload your own data".
    file_input = FileInput(accept=".csv", visible=False)
    # upload_status currently not used
    upload_status = Div(
        text="", styles={'color': "#247008", 'font-size': '13px'})
    # caches the last base64 payload (currently informational)
    _upload_buffer = {'data': None}

    def on_file_select_change(attr, old, new):
        """
        Toggle the upload widget visibility and, for pre-set systems,
        immediately load the CSV and run the AI Suggester so the sliders
        are pre-filled before the user even presses Train.
        """
        if new == "custom_upload":
            file_input.visible = True
            upload_status.text = "Please upload a CSV with columns: t, x1, x2..."
        else:
            file_input.visible = False
            path = os.path.join('data', new)
            if os.path.exists(path):
                df = pd.read_csv(path).astype(np.float64)
                apply_suggestion(df, f"<b>Selected system file: {new}</b>")
            else:
                upload_status.text = f"⚠ Pre-set file not found at {path}"

    file_select.on_change('value', on_file_select_change)

    def upload_to_local_drive(attr, old, new):
        """
        Callback fired when FileInput receives a new file. Bokeh delivers
        the file content as a base64 string in `new`. We decode it into a
        DataFrame purely to run the AI Suggester immediately (the actual
        training callback re-decodes file_input.value independently — see
        on_train_click — so this decode here is "preview only").
        """
        if not new:
            return
        _upload_buffer['data'] = new  # cache base64 payload
        try:
            decoded = base64.b64decode(new)
            f = io.BytesIO(decoded)
            df = pd.read_csv(f).astype(np.float64)
            apply_suggestion(df, "Custom file uploaded successfully!")
        except Exception as e:
            upload_status.text = f"⚠ Error processing uploaded file: {e}"

    file_input.on_change('value', upload_to_local_drive)

    # =========================================================================
    # SECTION 2 — MODEL CONFIGURATION CONTROLS
    # Train/validation split, Split type, Library type, polynomial degree, sparsity
    # threshold, and the Train/Delete button.
    # =========================================================================

    # Library select 
    library_select = Select(title="LIBRARY",
                            options=["Polynomial", "Fourier", "Combined"],
                            value="Polynomial")

    # Single slider controls the split; validation % is always 100 - train%.
    train_s = Slider(start=10, end=90, value=60, step=5,
                     title="Train/Validation Split")

    def on_train_s_change(attr, old, new):
        """Keep the human-readable split label in sync with the slider."""
        train_s.title = f"SPLIT: TRAIN {new}% | VALIDATION {100 - new}%"

    train_s.on_change('value', on_train_s_change)
    # Initialize title immediately
    on_train_s_change(None, None, train_s.value)
    train_s.show_value = False
    
    # Split type select
    split_select = Select(
        title="SPLIT STRATEGY",
        value="Random Sampling",
        options=[
            "Random Sampling",
            "Time-based",
            "Random Block",
        ],
        width=150,
    )

    # Degree/Harmonics slider
    poly_s = Slider(start=1, end=5,     value=1,
                    step=1,     title="DEGREE / HARMONICS")
    # Sparsity Threshold slider
    thr_s = Slider(start=0.0, end=0.5, value=0.1,
                   step=0.005, title="SPARSITY THRESHOLD")
    # ── Manual threshold input ──────────────────────────────────────────
    # Some systems turned out to be very sensitive to the exact threshold
    # value — the 0.005 slider step is too coarse for fine-tuning (e.g.
    # 0.0347 vs 0.035). This TextInput lets the user type an exact value;
    # it's two-way synced with thr_s so either control can drive the other.
    thr_input = TextInput(
        value=f"{thr_s.value:.4f}", title="Or type exact threshold:", width=150)

    # re-entrancy guard to prevent infinite update loops
    _thr_syncing = [False]

    def on_thr_slider_change(attr, old, new):
        """Slider moved → push the new value into the text box."""
        if _thr_syncing[0]:
            return
        _thr_syncing[0] = True
        thr_input.value = f"{new:.4f}"
        _thr_syncing[0] = False

    def on_thr_input_change(attr, old, new):
        """
        Text box edited → validate and push into the slider. Falls back
        silently to the last valid value if the typed text isn't a
        parseable, in-range number (e.g. mid-typing state like "0." or
        "-"), so the app never crashes on invalid manual input.
        """
        if _thr_syncing[0]:
            return
        try:
            val = float(new)
        except ValueError:
            return

        val = max(thr_s.start, min(thr_s.end, val))  # clamp to valid range

        _thr_syncing[0] = True
        thr_s.value = val
        thr_input.value = f"{val:.4f}"
        _thr_syncing[0] = False

    thr_s.on_change('value', on_thr_slider_change)
    thr_input.on_change('value', on_thr_input_change)
    
    # Train button
    btn_train = Button(label="TRAIN", button_type="primary",
                       height=50, width=100)

    # =========================================================================
    # SECTION 3 — HISTORY TABLE (LEADERBOARD)
    # Stores one row per training run with all metrics and the
    # discovered equations. Selecting a row re-renders that run's plots.
    # =========================================================================

    # Custom HTML template so multi-line equation strings wrap nicely
    # inside the DataTable cell instead of being clipped.
    eqn_template = """
    <div style="white-space: normal; word-wrap: break-word; line-height: 1.5;
                padding: 8px 0; font-family: 'Courier New', monospace;
                font-size: 12px; color: #00000;">
        <%= value %>
    </div>
    """
    eqn_formatter = HTMLTemplateFormatter(template=eqn_template)

    # Small HTML template for the merged Train/Val metric cells (Section 4
    # "compact" view) — packs R²/RMSE/MAE into 3 short lines instead of 3
    # separate wide columns, so the leaderboard needs a lot less horizontal
    # space per run.
    metrics_template = """
    <div style="white-space: normal; line-height: 1.4; padding: 4px 0;
                font-family: 'Courier New', monospace; font-size: 11px;">
        <%= value %>
    </div>
    """
    metrics_formatter = HTMLTemplateFormatter(template=metrics_template)

    def _fmt_metrics_html(label, color, r2, rmse, mae):
        return (
            f"<b style='color:{color};'>{label}</b><br>"
            f"R²: {r2:.4f}<br>RMSE: {rmse:.6f}<br>MAE: {mae:.6f}"
        )

    # "system" records which dataset/file produced the run (pre-set name or
    # the uploaded filename) so old runs stay interpretable at a glance.
    source_history = ColumnDataSource(data=dict(
        run=[], system=[], split=[], lib=[], poly=[], thr=[],
        train_metrics=[], val_metrics=[],
        rmse_diff=[], equations=[]
    ))

    # Train/Val each collapse into a single merged HTML cell (R²+RMSE+MAE
    # stacked) instead of 6 separate wide numeric columns.
    columns = [
        TableColumn(field="run",    title="Run #",      width=100),
        TableColumn(field="system", title="Data File",  width=400),
        TableColumn(field="split",  title="Split Type", width=300),
        TableColumn(field="lib",    title="Library",    width=200),
        TableColumn(field="poly",   title="Degree",     width=200),
        TableColumn(field="thr",    title="Threshold",  width=200),
        TableColumn(field="train_metrics", title="Train Metrics",
                    width=200, formatter=metrics_formatter),
        TableColumn(field="val_metrics",   title="Val Metrics",
                    width=200, formatter=metrics_formatter),
        TableColumn(field="rmse_diff", title="RMSE Diff", width=200),
        TableColumn(field="equations", title="Identified Equations",
                    width=1000, formatter=eqn_formatter),
    ]

    history_table = DataTable(
        source=source_history, columns=columns,
        width=1400, height=400, row_height=200,
        index_position=None, background="#ffffff",
        sortable=True, selectable=True
    )

    # Delete button
    btn_delete = Button(label="DELETE",
                        button_type="danger", width=100, height=50)
    btn_delete.disabled = True  # default disable when there is no run

    def on_row_select(attr, old, new):
        """
        Fired when the user clicks a row in the leaderboard. Re-renders
        the main result plot AND the diagnostic plots using the stored
        data for that run — this is what lets users "time travel" back
        to any previous run without re-training.
        """
        # when choose run -> appear button, allows user to delete runs
        btn_delete.disabled = not bool(new)
        if not new:
            return
        run_id = source_history.data['run'][new[0]]
        if run_id in trained_model_storage:
            render_plot(run_id)
            diag = trained_model_storage[run_id].get('diagnostics')
            if diag:
                _render_diag_plots(diag)

    source_history.selected.on_change('indices', on_row_select)

    # =========================================================================
    # SECTION 4 — MAIN RESULT PLOT
    # Shows train/validation points scattered against the SINDy-simulated
    # trajectory for the currently-viewed run.
    # =========================================================================

    # Main plot
    p = figure(title="Model Result", height=430, sizing_mode="stretch_width")
    # Empty invisible glyph forces Bokeh to allocate a renderer/legend slot
    # immediately, avoiding a "plot has zero renderers" warning on first load.
    p.scatter([], [], alpha=0)
    p.legend.click_policy = "hide"
    
    # Hovertool only applies to SINDy fit line, not for train/validation points
    # mode:
    # vline: whenever a vertical line from the mouse position intersects a glyph
    # hline: whenever a horizontal line from the mouse position intersects a glyph
    # mouse: only when the mouse is directly over a glyph
    # currently set vline to compare position accross all the states
    fit_hover = HoverTool(
        renderers=[], # default: no SINDy fit line has been drawn, when call render_plot() will be modified
        mode="vline",
        tooltips=[
            ("Variable", "@name"),
            ("t", "@t{0.000}"),
            ("Value", "@y{0.0000}"),],
    )
    p.add_tools(fit_hover)

    # Storage for main-plot renderers, keyed by state index, so the toggle
    # callbacks below can reach in and adjust alpha per (state, role) pair.
    # Populated fresh each render_plot() call.
    _main_renderers = {}   # {state_idx: {'train': renderer, 'val': renderer, 'fit': renderer}}

    # Two independent toggle groups — this is why we don't use Bokeh's native
    # legend click_policy here: a single legend can only group renderers along
    # ONE axis (either "by state" or "by role"), but we want both axes toggled
    # independently (e.g. mute state x2 AND separately hide all fit lines).
    state_toggle = CheckboxButtonGroup(
        labels=[], active=[], button_type="default")
    layer_toggle = CheckboxButtonGroup(
        labels=["Data points", "SINDy fit"], active=[0, 1], button_type="default")

    # Static color key (state name -> color) since state_toggle button labels
    # are plain text and can't carry per-button color — this Div is the visual
    # reference, the buttons next to it are what actually drive visibility.
    state_key_div = Div(text="", styles={'padding': '2px 0'})

    def _update_main_visibility(attr, old, new):
        """
        Recompute alpha for every renderer on the main plot as the AND of
        (state selected in state_toggle) and (its role selected in
        layer_toggle). Fading (not full hide) so a de-selected state stays
        spatially legible relative to the ones still highlighted.
        """
        active_states = set(state_toggle.active)
        data_on = 0 in set(layer_toggle.active)
        fit_on = 1 in set(layer_toggle.active)

        for i, rends in _main_renderers.items():
            state_on = i in active_states
            
            # Currently hide, if want to fade change 0 to 0.02
            train_alpha = 0.35 if (state_on and data_on) else 0
            val_alpha = 0.55 if (state_on and data_on) else 0
            fit_alpha = 1.0 if (state_on and fit_on) else 0

            rends['train'].glyph.fill_alpha = train_alpha
            rends['train'].glyph.line_alpha = train_alpha
            rends['val'].glyph.fill_alpha = val_alpha
            rends['val'].glyph.line_alpha = val_alpha
            if rends['fit'] is not None:
                rends['fit'].glyph.line_alpha = fit_alpha

    state_toggle.on_change('active', _update_main_visibility)
    layer_toggle.on_change('active', _update_main_visibility)

    # This div is currently not used.
    # Previously, this div is to return equation found by SINDy, otherwise return a error message
    # Since equation is displayed in the leaderboard, I made a new user_warning_div below serving as
    # a status update: train complete / warning message
    res_div = Div(
        text="<h3>Run Equations:</h3>",
        styles={'background': '#f8f9fa',
                'padding': '10px', 'border-radius': '5px'}
    )

    # =========================================================================
    # SECTION 5 — RESIDUAL DIAGNOSTIC PLOTS
    # Three complementary views of model quality on the derivative (dX/dt)
    # space, used to spot missing library terms or structured (non-random)
    # error that a single R²/RMSE number would hide.
    # =========================================================================

    # Plot 1: Residual vs Time
    # A well-fit model's residuals should look like structureless noise.
    # Visible trends/oscillations indicate the model is missing a term.
    p_resid = figure(
        title="Residual vs Time",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="Time",
        y_axis_label="Residual",
        toolbar_location=None,
    )
    p_resid.scatter([], [], alpha=0)

    # Plot 2: FFT of Residual
    # A dominant frequency peak in the residual spectrum means there is
    # still periodic structure in the error → the candidate library is
    # missing a sin/cos (or higher harmonic) term.
    p_fft = figure(
        title="Residual FFT (Frequency Content)",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="Frequency (Hz)",
        y_axis_label="Amplitude",
        toolbar_location=None,
    )
    p_fft.scatter([], [], alpha=0)

    # Plot 3: dX_true vs dX_predicted scatter
    # A perfect model places every point exactly on the y=x diagonal.
    # Systematic curvature/fanning indicates bias or heteroscedastic error.
    p_scatter = figure(
        title="dX True vs dX Predicted",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="dX Predicted",
        y_axis_label="dX True",
        toolbar_location=None,
    )
    p_scatter.scatter([], [], alpha=0)

    # Text summary shown above the 3 diagnostic plots (R², SNR, autocorrelation
    # per state variable).
    diag_stats_div = Div(
        text="<a>Run a training session to see diagnostics.</a>",
        styles={'font-size': '13px', 'color': '#00000'}
    )

    counter = [0]   # run counter — monotonically increasing, never reset
    # even after deletions (see project history: run IDs
    # are intentionally permanent to avoid ambiguity).
    
    # This is the user_warning_div that I talked about just above section 5
    user_warning_div = Div(
        text="",
        styles={'color': '#7f8c8d', 'font-size': '13px', 'padding': '4px 0'}
    )
    # Tracks which run_id is currently displayed on the main plot — used by
    # on_delete_click to decide whether to clear the plot, now that
    # user_warning_div's text no longer always contains "Run #{run_id}".
    _current_view_run = [None]
    # Static role-legend — separate from Bokeh's interactive per-state legend.
    # Needed because merging train/val/fit under one legend_label per state
    # (for mute-by-variable) removes the old separate "Train points"/"Val
    # points" entries, so marker-shape meaning must be spelled out explicitly
    # somewhere the student can't miss — this is pedagogically load-bearing
    # since the whole point of this plot is showing HOW the split partitions
    # the data differently per split strategy.

    def render_plot(run_id):
        """
        Redraw the main result plot from the stored plot_data of a given run.
        Overlays all state variables. Visibility is driven entirely by
        state_toggle / layer_toggle (see _update_main_visibility) — no
        Bokeh legend interactivity on this plot.
        """
        data = trained_model_storage[run_id]['plot_data']
        t, X = data['t'], data['X']
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        x_sim_full = data['x_sim']
        names = trained_model_storage[run_id].get('feature_names') or \
            [f"x{i+1}" for i in range(X.shape[1])]

        p.renderers = []
        _main_renderers.clear()

        n_vars = X.shape[1]
        color_key_parts = []

        for i in range(n_vars):
            color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
            label = names[i] if i < len(names) else f"x{i+1}"

            # Data points stay neutral gray — color is reserved for the fit
            # line only, so a well-fit curve never gets visually swallowed
            # by same-colored data points (see earlier fix).
            r_train = p.scatter(t[train_idx], X[train_idx, i],
                                color="#1f77b4", alpha=0.35, size=4, legend_label="Train points")
            r_val = p.scatter(t[val_idx], X[val_idx, i],
                              color="#ff7f0e", alpha=0.55, size=4, legend_label="Val points")
            r_fit = None
            if x_sim_full is not None:
                r_fit = p.line(t, x_sim_full[:, i],
                               color=color, line_width=2.8)

            _main_renderers[i] = {'train': r_train, 'val': r_val, 'fit': r_fit}
            color_key_parts.append(
                f"<span style='color:{color}; font-weight:700;'>●</span> "
                f"<span style='color:#2c3e50;'>{label}</span>"
            )
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"

        state_key_div.text = (
            "<div style='font-size:14px;'>" +
            "&nbsp;&nbsp;".join(color_key_parts) + "</div>"
        )

        # Re-sync the two toggle groups to this run: fresh labels, everything
        # visible by default.
        state_toggle.labels = names[:n_vars] if len(names) >= n_vars else \
            [f"x{i+1}" for i in range(n_vars)]
        state_toggle.active = list(range(n_vars))
        layer_toggle.active = [0, 1]
        # apply default alphas immediately
        _update_main_visibility(None, None, None)

        p.title.text = f"Model Result — Run #{run_id}"
        _current_view_run[0] = run_id

        # user_warning_div: shows the fit warning for this run (e.g.
        # sparsity threshold too high -> all coefficients eliminated) if
        # one was recorded, otherwise just confirms the run trained fine.
        warning_msg = trained_model_storage[run_id].get('warning')
        if warning_msg:
            user_warning_div.text = f"<b style='color:#d91212;'>⚠ {warning_msg}</b>"
        else:
            user_warning_div.text = "<b style='color:#27ae60;'>✅ Train complete</b>"

    # Shared color palette for multi-variable diagnostic plots (cycles if
    # a system has more than 5 state variables).
    _DIAG_COLORS = ["#61e0ee", "#ebc626", "#2ca02c", "#d62728", "#9467bd"]

    def _render_diag_plots(diag):
        """
        Populate the 3 diagnostic plots (Section 6) from a diagnostics dict
        produced by engine.compute_diagnostics().

        Expected `diag` structure:
            diag['t']            -> time array
            diag['residuals']    -> {var_name: residual array}
            diag['fft_freqs']    -> frequency axis (shared across variables)
            diag['fft_amps']     -> {var_name: FFT amplitude array}
            diag['dX_true']      -> {var_name: true derivative array}
            diag['dX_pred']      -> {var_name: predicted derivative array}
            diag['stats']        -> {var_name: {r2_dx, snr_db, autocorr}}

        FFT x-axis auto-scaling
        ------------------------
        We auto-scale the residual FFT x-axis to the frequency range that
        actually contains meaningful energy. This avoids two problems:
          1. Hardcoded ranges that only work for one specific system.
          2. Showing the full Nyquist range where most content is noise
             floor, making real peaks hard to see.
        """
        if not diag:
            return

        # Clear all 3 plots before redrawing.
        p_resid.renderers = []
        p_fft.renderers = []
        p_scatter.renderers = []
        if p_resid.legend:
            p_resid.legend.items = []
        if p_fft.legend:
            p_fft.legend.items = []
        if p_scatter.legend:
            p_scatter.legend.items = []

        var_names = list(diag['residuals'].keys())
        freqs = diag['fft_freqs']

        for idx, name in enumerate(var_names):
            color = _DIAG_COLORS[idx % len(_DIAG_COLORS)]

            # Plot 1 — Residual vs Time
            p_resid.line(
                diag['t'], diag['residuals'][name],
                color=color, line_width=1.5, alpha=0.8,
                legend_label=name,
                muted_color=color, muted_alpha=0.12,
            )

            # Plot 2 — FFT amplitude spectrum of the residual
            p_fft.line(
                freqs, diag['fft_amps'][name],
                color=color, line_width=1.5, alpha=0.8,
                legend_label=name,
                muted_color=color, muted_alpha=0.12,
            )

            # Plot 3 — dX_true vs dX_pred scatter
            p_scatter.scatter(
                diag['dX_pred'][name], diag['dX_true'][name],
                color=color, alpha=0.3, size=4,
                legend_label=name,
                muted_color=color, muted_alpha=0.06,
            )

        # ── Auto-scale FFT x-axis ──────────────────────────────────────────
        # Combine amplitude across all variables to find the global energy
        # envelope, then show only the range where at least one variable
        # has meaningful energy (> 1% of the global peak amplitude). This
        # generalizes to any system — slow biological oscillators, fast
        # mechanical systems, chaotic attractors — without any hardcoded
        # frequency limit.
        all_amps = np.concatenate([diag['fft_amps'][n] for n in var_names])
        max_amp = float(all_amps.max())

        if max_amp > 0:
            significant_indices = np.where(all_amps > 0.01 * max_amp)[0]

            if len(significant_indices) > 0:
                # all_amps is a concatenation of n_vars arrays each of
                # length n_freqs — map the flat index back onto the shared
                # frequency axis with a modulo.
                n_freqs = len(freqs)
                last_idx = int(significant_indices[-1]) % n_freqs
                f_max = float(freqs[last_idx])

                # 20% margin so the last visible peak isn't clipped at the edge.
                p_fft.x_range.end = f_max * 1.2
                p_fft.x_range.start = 0.0

        # Add a y=x reference line to Plot 3 (the "ideal fit" diagonal).
        all_vals = np.concatenate([diag['dX_true'][n] for n in var_names])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        p_scatter.line(
            [vmin, vmax], [vmin, vmax],
            color="#e74c3c", line_width=1.5, line_dash="dashed",
            legend_label="ideal",
        )

        for fig in [p_resid, p_fft, p_scatter]:
            fig.legend.click_policy = "mute"
            fig.legend.location = "top_right"

        # Build the stats summary — one line per state variable.
        stats_html = "<b>Residual Stats:</b><br>"
        for name, s in diag['stats'].items():
            stats_html += (
                f"&nbsp;&nbsp;<b>{name}</b>: "
                f"R²(dX)={s['r2_dx']} | "
                f"SNR={s['snr_db']} dB | "
                f"autocorr={s['autocorr']}<br>"
            )
        stats_html += (
            "<span style='color:#7f8c8d; font-size:11px;'>"
            "<i>Computed on the full dataset (train + validation combined) — "
            "not directly comparable to the Train/Val R² in the leaderboard below.</i>"
            "</span>"
        )
        diag_stats_div.text = stats_html

    # =========================================================================
    # SECTION 6 — TRAIN CALLBACK
    # Main entry point triggered by the "TRAIN" button. Loads data (pre-set
    # or uploaded), fits SINDy on a random split, computes diagnostics,
    # updates the leaderboard, and re-renders all plots.
    # =========================================================================

    def on_train_click():
        # ── 1. Resolve the data source (pre-set file vs uploaded CSV) ──────
        is_custom = (file_select.value == "custom_upload")
        uploaded_value = None
        if is_custom:
            try:
                uploaded_value = file_input.value
            except Exception:
                uploaded_value = None

        if is_custom:
            if not uploaded_value:
                # Guard against pressing Train before a file was actually chosen.
                res_div.text = "<span style='color:red;'>⚠ Please upload a CSV file first!</span>"
                return
            # Decode the uploaded CSV (base64 -> bytes -> DataFrame).
            decoded = base64.b64decode(file_input.value)
            f = io.BytesIO(decoded)
            df = pd.read_csv(f).astype(np.float64)
            # FileInput.filename is only populated in newer Bokeh versions —
            # fall back to a generic label so the leaderboard never shows blank.
            data_file_label = getattr(
                file_input, 'filename', None) or "Custom Upload"
        else:
            # Load one of the bundled pre-set system files.
            path = os.path.join('data', file_select.value)
            df = pd.read_csv(path).astype(np.float64)
            data_file_label = _SYSTEM_LABELS.get(
                file_select.value, file_select.value)

        counter[0] += 1  # unique, ever-increasing run ID

        # ── 2. Parse data into time / state matrices ────────────────────────
        t = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        names = list(df.columns[1:])
        train_frac = train_s.value / 100.0

        # ── 3. Fit SINDy on a random train/validation split ─────────────────
        # Derivatives are computed once on the FULL trajectory, then the
        # resulting (X, dX) pairs are split randomly — this is more robust
        # than splitting the raw time series first, because finite-difference
        # derivatives near a split boundary would otherwise be biased.
        fit_warning_msg = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model, train_idx, val_idx, m_train, m_val = \
                    engine.fit_model(
                        X, t,
                        poly_degree=poly_s.value,
                        threshold=thr_s.value,
                        names=names,
                        lib_type=library_select.value,
                        train_frac=train_frac,
                        random_seed=counter[0] * 7,  # unique seed per run
                        split_method=split_select.value.lower()
                    )
                if caught:
                    # keep the last warning message, as it's the most important
                    fit_warning_msg = str(caught[-1].message)
        except Exception as e:
            res_div.text = f"<span style='color:red;'>⚠ Fit error: {e}</span>"
            return

        # ── 4. Compute residual diagnostics for the 3 diagnostic plots ──────
        diag = engine.compute_diagnostics(X, t)

        t_r2, t_rmse, t_mae = m_train['r2'], m_train['rmse'], m_train['mae']
        v_r2, v_rmse, v_mae = m_val['r2'],   m_val['rmse'],   m_val['mae']
        rmse_diff = float(np.abs(t_rmse - v_rmse))

        # ── 5. Forward-simulate the discovered equations over the full
        #        time range for visualization (x(t) reconstruction from
        #        the initial condition, NOT the raw dX/dt fit) ─────────────
        try:
            x_sim_full = engine.simulate(X[0], t)
        except Exception as e:
            res_div.text = f"<span style='color:red;'>⚠ Simulation error: {e}</span>"
            return

        # ── 6. Format the discovered equations for display ─────────────────
        raw_eqs = engine.get_equations()
        formatted_eqs_html = "".join(
            [f"<b style='color:#e74c3c;'>({i+1})</b> {eq}<br>" for i,
             eq in enumerate(raw_eqs)]
        )

        # ── 7. Append a new row to the leaderboard ──────────────────────────
        new_entry = {
            'run':        [counter[0]],
            'system':     [data_file_label],
            'split':      [split_select.value],
            'lib':        [library_select.value],
            'poly':       [poly_s.value],
            'thr':        [thr_s.value],
            'train_metrics': [_fmt_metrics_html("TRAIN", "#1f77b4", t_r2, t_rmse, t_mae)],
            'val_metrics':   [_fmt_metrics_html("VAL", "#ff7f0e", v_r2, v_rmse, v_mae)],
            'rmse_diff':  [f"{rmse_diff:.6f}"],
            'equations':  [formatted_eqs_html],
        }
        source_history.stream(new_entry)

        # ── 8. Persist everything needed to reconstruct this run later ─────
        # (used by render_plot/_render_diag_plots on row-select, and by the
        # Test/Predict tabs to simulate from a saved model instance).
        trained_model_storage[counter[0]] = {
            'run_id':             counter[0],
            'system_name':        file_select.value,
            'split_strategy':     split_select.value,
            # snapshot — engine.model gets overwritten on next Train
            'model_instance':     copy.deepcopy(engine.model),
            'lib_type':           library_select.value,
            'poly_degree':        poly_s.value,
            'threshold':          thr_s.value,
            'feature_names':      names,             # variable names from the CSV header
            'initial_conditions': X[0].tolist(),
            'metrics': {
                'train_rmse': t_rmse,
                'val_rmse':   v_rmse,
                'rmse_diff':  rmse_diff,
                'val_r2':     v_r2,
            },
            'equations':  raw_eqs,
            'warning':    fit_warning_msg,
            'plot_data': {
                't':         t,
                'X':         X,
                'train_idx': train_idx,
                'val_idx':   val_idx,
                'x_sim':     x_sim_full,
            },
            'diagnostics': diag,
        }

        # ── 9. Render the plots for the run that was just trained ──────────
        render_plot(counter[0])
        _render_diag_plots(diag)

    def on_delete_click():
        """
        Remove the currently-selected leaderboard row: deletes the model
        from trained_model_storage, removes the row from the DataTable,
        and clears the main/diagnostic plots if the deleted run was the
        one currently being viewed.

        NOTE: Run IDs (`counter`) are intentionally NOT reset/renumbered
        after a deletion — every run ID stays permanently unique so past
        references (e.g. from the Test/Predict tabs) never become ambiguous.
        """
        selected = source_history.selected.indices
        if not selected:
            return

        idx = selected[0]
        run_id = source_history.data['run'][idx]

        # Remove from the model storage dict.
        if run_id in trained_model_storage:
            del trained_model_storage[run_id]

        # Remove the row from the DataTable by rebuilding every column list
        # with that index filtered out (ColumnDataSource has no native
        # "delete row" API).
        new_data = {k: [v for i, v in enumerate(vals) if i != idx]
                    for k, vals in source_history.data.items()}
        source_history.data = new_data
        source_history.selected.indices = []

        # If the deleted run was the one currently displayed, clear the
        # main result plot back to an empty state.
        if _current_view_run[0] == run_id:
            p.renderers = []
            p.title.text = "Model Result"
            user_warning_div.text = ""
            _current_view_run[0] = None
            _main_renderers.clear()
            state_toggle.labels = []
            state_toggle.active = []
            state_key_div.text = ""

        # Clear all 3 diagnostic plots too.
        for figs in [p_resid, p_fft, p_scatter]:
            figs.renderers = []
            if figs.legend and len(figs.legend) > 0:
                figs.legend[0].items = []

        # Reset stats text and FFT x-axis range back to neutral defaults —
        # the range will be auto-scaled again on the next training run.
        diag_stats_div.text = "<i>Run a training session to see diagnostics.</i>"
        p_fft.x_range.start = 0.0
        p_fft.x_range.end = 1.0

    btn_delete.on_click(on_delete_click)

    # ── Run the AI Suggester once on page load for the default pre-set
    #     system, so the sliders aren't left at arbitrary defaults before
    #     the user has interacted with anything. ──────────────────────────
    # div apply_suggestion is currently not used, but still suggest hyperparameter when choose data file.
    initial_path = os.path.join('data', file_select.value)
    if os.path.exists(initial_path):
        try:
            df_init = pd.read_csv(initial_path).astype(np.float64)
            apply_suggestion(
                df_init, f"<b>Loaded default pre-set system: {file_select.value}</b>")
        except Exception:
            pass  # non-fatal — user can still configure manually

    btn_train.on_click(on_train_click)

    # =========================================================================
    # SECTION 7 — LAYOUT ASSEMBLY
    # =========================================================================

    top_row = row(
        column(file_select, file_input, train_s, split_select, library_select,
               poly_s, thr_s, thr_input, row(btn_train, btn_delete), user_warning_div, width=320),
        column(p, 
               # the row below is to align: "center", but since bokeh doesnt have that css style, so we use Spacer instead
               row(Spacer(sizing_mode="stretch_width"), state_key_div, Spacer(sizing_mode="stretch_width"), sizing_mode="stretch_width"),
               row(Spacer(sizing_mode="stretch_width"), row(state_toggle, layer_toggle), Spacer(sizing_mode="stretch_width"), sizing_mode="stretch_width"),
                sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )

    return column(
        top_row,
        Div(text="<hr><b>RESIDUAL DIAGNOSTICS</b>"),
        diag_stats_div,
        row(p_resid, p_fft, p_scatter, sizing_mode="stretch_width"),
        Div(text="<hr><b>TRAINING HISTORY — Metrics on dx/dt (derivative space)</b>"),
        history_table,
        sizing_mode="stretch_width"
    )