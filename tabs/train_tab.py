# tabs/train_tab.py

from bokeh.models import (ColumnDataSource, Slider, Div, Button,
                          Select, DataTable, TableColumn, HTMLTemplateFormatter, FileInput)
from bokeh.layouts import column, row
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import os
import copy
import base64
import io


def train_tab_layout(engine, trained_model_storage):
    
    def analyze_data_linearity(df):
        try:
            t      = df.iloc[:, 0].values
            X      = df.iloc[:, 1:].values
            n_vars = X.shape[1]
            dt     = np.mean(np.diff(t)) if len(t) > 1 else 0.1

            # ── 1. Calculate derivative ──────────────────────────────
            # use Savitzky-Golay 
            from scipy.signal import savgol_filter
            dXdt = np.zeros_like(X)
            window = min(11, len(t) // 10 * 2 + 1) 
            window = max(window, 5)
            for i in range(n_vars):
                smoothed = savgol_filter(X[:, i], window_length=window, polyorder=3)
                dXdt[:, i] = np.gradient(smoothed, dt)

            # ── 2. Compare R² of degree 1, 2, 3 ────────────────────────
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            r2_scores = {}
            for deg in [1, 2, 3]:
                poly   = PolynomialFeatures(degree=deg, include_bias=True)
                X_poly = poly.fit_transform(X)
                lr     = LinearRegression(fit_intercept=False).fit(X_poly, dXdt)
                r2_scores[deg] = r2_score(dXdt, lr.predict(X_poly),
                                        multioutput='uniform_average')

            r2_linear = r2_scores[1]
            r2_deg2   = r2_scores[2]
            r2_deg3   = r2_scores[3]

            # ── 3. Choose minimal degree that satisfy R² ──────────────────
            # If degree=1 is sufficient → linear system
            if r2_linear >= 0.85:
                sug_degree = 1
                reason_deg = f"Linear fit R²={r2_linear:.3f} ≥ 0.85 → degree=1"
            # If degree=2 is significantly larger than degree=1
            elif r2_deg2 - r2_linear >= 0.05:
                sug_degree = 2
                reason_deg = (f"Degree=2 R²={r2_deg2:.3f} improves over "
                              f"linear R²={r2_linear:.3f} by {r2_deg2 - r2_linear:.3f} → degree=2")
            # If degree=3 is significantly larger than degree=2
            elif r2_deg3 - r2_deg2 >= 0.05:
                sug_degree = 3
                reason_deg = (f"Degree=3 R²={r2_deg3:.3f} improves over "
                              f"degree=2 R²={r2_deg2:.3f} by {r2_deg3 - r2_deg2:.3f} → degree=3")
            # No significant difference → linear
            else:
                sug_degree = 1
                reason_deg = (f"No significant improvement beyond linear "
                              f"(linear={r2_linear:.3f}, deg2={r2_deg2:.3f}, deg3={r2_deg3:.3f}) → degree=1")

            # ── 4. FFT — detect periodicity ──────────────────────────────
            is_periodic = False
            for i in range(n_vars):
                signal   = X[:, i] - np.mean(X[:, i])
                fft_vals = np.abs(np.fft.rfft(signal))
                peaks    = fft_vals[1:]
                if len(peaks) > 0:
                    if np.max(peaks) > 5 * np.mean(peaks):
                        is_periodic = True
                        break

            # ── 5. Noise estimate → suggest threshold ────────────────────
            noise_estimates = []
            for i in range(n_vars):
                amp  = np.abs(np.fft.rfft(X[:, i])) / len(t)
                high = np.sort(amp)[-max(1, int(len(amp) * 0.2)):]
                noise_estimates.append(float(np.median(high)))
            noise_level = float(np.mean(noise_estimates))

            if noise_level < 0.01:
                sug_threshold = 0.05
            elif noise_level < 0.05:
                sug_threshold = 0.10
            else:
                sug_threshold = 0.20

            # ── 6. Library (Currently not used) ────────────────────────────────────
            if is_periodic and r2_linear < 0.92:
                sug_library = "Combined"
                reason_lib  = "Periodic signal + nonlinear → Try Polynomial/Fourier/"
            elif is_periodic and r2_linear >= 0.92:
                sug_library = "Fourier"
                reason_lib  = "Clear periodic signal + linear → Fourier"
            else:
                sug_library = "Polynomial"
                reason_lib  = "No dominant periodicity → Polynomial"

            reason = f"{reason_lib}. {reason_deg}. Threshold={sug_threshold} (noise≈{noise_level:.4f})."
            return sug_library, sug_degree, sug_threshold, reason

        except Exception as e:
            return "Polynomial", 1, 0.10, f"Error analyzing data: {e}"
    
    # helper function to take the result from the function above and apply it to the UI
    def apply_suggestion(df, prefix_msg= ""):
        # call the function above
        lib, deg, thr, reason = analyze_data_linearity(df)
        # change the value of parameters to the suggestion
        poly_s.value = deg
        thr_s.value = thr
        # print the reason to the UI
        upload_status.text = f"{prefix_msg}<br><b style='color:#e67e22;'>🤖 AI Suggester:</b> {reason}"
    
    # 1. UI Components
    system_options = [
        ("cs_train_data.csv", "Coupled Spring-Mass (Pre-set)"),
        ("vanderpol_train.csv", "Van der Pol Oscillator (Pre-set)"),
        ("custom_upload", "Upload your own data")
    ]
    
    file_select = Select(title="1) SELECT SYSTEM", options=system_options,
                         value="cs_train_data.csv")

    # Widget Upload (hidden as default)
    file_input = FileInput(accept=".csv", visible=False)
    upload_status = Div(text="", styles={'color': '#e67e22', 'font-size': '12px'})
    _upload_buffer = {'data': None}

    def on_file_select_change(attr, old, new):
        if new == "custom_upload":
            file_input.visible = True
            upload_status.text = "ℹ️ Please upload a CSV with columns: t, x1, x2..."
        else:
            file_input.visible = False
            path = os.path.join('data', new)
            if os.path.exists(path):
                df = pd.read_csv(path).astype(np.float64)
                # Activate suggestion function when choose preset system
                apply_suggestion(df, f"✅ Selected system file: <b>{new}</b>")
            else:
                upload_status.text = f"⚠ Pre-set file not found at {path}"
    
    file_select.on_change('value', on_file_select_change)
    
    def upload_to_local_drive(attr, old, new):
        if not new: return
        _upload_buffer['data'] = new  # cache base64
        try:
            # decode data file just uploaded
            decoded = base64.b64decode(new)
            f = io.BytesIO(decoded)
            df = pd.read_csv(f).astype(np.float64)
            
            # activate data analysis function
            apply_suggestion(df, "✅ Custom file uploaded successfully!")
        except Exception as e:
            upload_status.text = f"⚠ Error processing uploaded file: {e}"
    file_input.on_change('value', upload_to_local_drive)
    
    library_select = Select(title="2) LIBRARY",
                            options=["Polynomial", "Fourier", "Combined"],
                            value="Polynomial")

    # 2 slider Train / Validation
    train_s = Slider(start=10, end=90, value=60, step=5,
                     title="Train - Validation Split")
 
    split_div = Div(
        text="<b style='color:#27ae60;'>✅ Split: Train 60% | Val 40%</b>",
        styles={'padding': '4px 0'}
    )
 
    def on_train_s_change(attr, old, new):
        split_div.text = (
            f"<b style='color:#27ae60;'>✅ "
            f"Train {new}% | Val {100 - new}%</b>"
        )
    
    train_s.on_change('value', on_train_s_change)

    poly_s = Slider(start=1, end=5,     value=1,    step=1,     title="Degree / Harmonics")
    thr_s  = Slider(start=0.001, end=0.5, value=0.1, step=0.005, title="Sparsity Threshold")
    btn_train = Button(label="TRAIN", button_type="success", height=50)

    # -------------------------------------------------------------------------
    # History Table
    # -------------------------------------------------------------------------
    eqn_template = """
    <div style="white-space: normal; word-wrap: break-word; line-height: 1.5;
                padding: 8px 0; font-family: 'Courier New', monospace;
                font-size: 12px; color: #2c3e50;">
        <%= value %>
    </div>
    """
    eqn_formatter = HTMLTemplateFormatter(template=eqn_template)

    source_history = ColumnDataSource(data=dict(
        run=[], lib=[], poly=[], thr=[],
        train_r2=[], train_rmse=[], train_mae=[],
        val_r2=[],   val_rmse=[],   val_mae=[],
        rmse_diff=[], equations=[]
    ))

    columns = [
        TableColumn(field="run",        title="Run #",           width=200),
        TableColumn(field="lib",        title="Library",         width=300),
        TableColumn(field="train_r2",   title="Train R² (dX)",   width=200),
        TableColumn(field="train_rmse", title="Train RMSE (dX)", width=200),
        TableColumn(field="train_mae",  title="Train MAE (dX)",  width=200),
        TableColumn(field="val_r2",     title="Val R² (dX)",     width=200),
        TableColumn(field="val_rmse",   title="Val RMSE (dX)",   width=200),
        TableColumn(field="val_mae",    title="Val MAE (dX)",    width=200),
        TableColumn(field="rmse_diff",  title="RMSE Diff",       width=200),
        TableColumn(field="equations",  title="Identified Equations",
                    width=1000, formatter=eqn_formatter),
    ]

    history_table = DataTable(
        source=source_history, columns=columns,
        width=1200, height=400, row_height=200,
        index_position=None, background="#ffffff",
        sortable=True, selectable=True,
    )
    def on_row_select(attr, old, new):
        if not new:
            return
        run_id = source_history.data['run'][new[0]]
        if run_id in trained_model_storage:
            render_plot(run_id)
    source_history.selected.on_change('indices', on_row_select)

    # -------------------------------------------------------------------------
    # 3. Plot
    # -------------------------------------------------------------------------
    p = figure(title="Model Result",
               width=850, height=400)
    p.legend.click_policy = "hide"

    res_div = Div(
        text="<h3>Run Equations:</h3>",
        styles={'background': '#f8f9fa', 'padding': '10px', 'border-radius': '5px'}
    )

    counter = [0]
    view_div = Div(
            text="",
            styles={'color': '#7f8c8d', 'font-size': '13px', 'padding': '4px 0'}
        )

    def render_plot(run_id): # rewrite plotting function to allow user see the plot when press on any run
        """Redraw plot from stored plot_data of a given run."""
        data = trained_model_storage[run_id]['plot_data']
        t, X         = data['t'], data['X']
        train_idx    = data['train_idx']
        val_idx      = data['val_idx']
        x_sim_full   = data['x_sim']

        p.renderers = []
        if p.legend:
            p.legend.items = []
        # Train points in BLUE
        p.scatter(t[train_idx], X[train_idx, 0],
                color="#1f77b4", alpha=0.4, size=4, legend_label="Train points")
        # Validation points in ORANGE
        p.scatter(t[val_idx], X[val_idx, 0],
                color="#ff7f0e", alpha=0.4, size=4, legend_label="Val points")
        if x_sim_full is not None:
            p.line(t, x_sim_full[:, 0],
                color="#2ecc71", line_width=2.5, legend_label="SINDy found")

        p.legend.click_policy = "hide"
        p.legend.location     = "top_right"
        p.title.text = f"Model Result — Run #{run_id}"
        view_div.text = f"<b style='color:#2c3e50;'>👁 Viewing Run #{run_id}</b>"

    # -------------------------------------------------------------------------
    # 4. Callback
    # -------------------------------------------------------------------------
    def on_train_click():
        #1 : check if user choose custom upload or preset system
        is_custom = (file_select.value == "custom_upload")
        uploaded_value = None
        if is_custom: # user choose custom upload
            try:
                uploaded_value = file_input.value
            except Exception:
                uploaded_value = None
        
        if is_custom:
            if not uploaded_value: # user choose custom but not upload data file before press train -> print out a warning message
                res_div.text = "<span style='color:red;'>⚠ Please upload a CSV file first!</span>"
                return
            # read from the data file uploaded
            # 1. get decoded data from upload
            decoded = base64.b64decode(file_input.value)
            # 2. make it a stream byte
            f = io.BytesIO(decoded)
            # 3. read data straigth from the file
            df = pd.read_csv(f).astype(np.float64)
        else: #read from preset files
            path      = os.path.join('data', file_select.value)
            df        = pd.read_csv(path).astype(np.float64)

        counter[0] += 1

        # Load data
        t         = df.iloc[:, 0].values
        X         = df.iloc[:, 1:].values
        names     = list(df.columns[1:])
        train_frac = train_s.value / 100.0

        # Random split + fit
        try:
            model, train_idx, val_idx, m_train, m_val = \
                engine.fit_model_random_split(
                    X, t,
                    poly_degree  = poly_s.value,
                    threshold    = thr_s.value,
                    names        = names,
                    lib_type     = library_select.value,
                    train_frac   = train_frac,
                    random_seed  = counter[0] * 7,  # different seed each run
                )
        except Exception as e:
            res_div.text = f"<span style='color:red;'>⚠ Fit error: {e}</span>"
            return

        t_r2   = m_train['r2'];   t_rmse = m_train['rmse']; t_mae = m_train['mae']
        v_r2   = m_val['r2'];     v_rmse = m_val['rmse'];   v_mae = m_val['mae']
        rmse_diff = float(np.abs(t_rmse - v_rmse))

        # Simulate to plot — use all time array
        try:
            x_sim_full = engine.simulate(X[0], t)
        except Exception as e:
            res_div.text = f"<span style='color:red;'>⚠ Simulation error: {e}</span>"
            return

        # Format equations
        raw_eqs = engine.get_equations()
        formatted_eqs_html = "".join(
            [f"<b style='color:#e74c3c;'>({i+1})</b> {eq}<br>" for i, eq in enumerate(raw_eqs)]
        )

        # Stream to leaderboard
        new_entry = {
            'run':        [counter[0]],
            'lib':        [library_select.value],
            'poly':       [poly_s.value],
            'thr':        [thr_s.value],
            'train_r2':   [f"{t_r2:.4f}"],
            'train_rmse': [f"{t_rmse:.6f}"],
            'train_mae':  [f"{t_mae:.6f}"],
            'val_r2':     [f"{v_r2:.4f}"],
            'val_rmse':   [f"{v_rmse:.6f}"],
            'val_mae':    [f"{v_mae:.6f}"],
            'rmse_diff':  [f"{rmse_diff:.6f}"],
            'equations':  [formatted_eqs_html],
        }
        source_history.stream(new_entry)

        # save model to storage
        trained_model_storage[counter[0]] = {
            'run_id':         counter[0],
            'system_name': file_select.value,
            'model_instance': copy.deepcopy(engine.model),
            'lib_type':       library_select.value,
            'poly_degree':    poly_s.value,
            'threshold':      thr_s.value,
            'feature_names': names, #save variable names from csv file
            'initial_conditions' : X[0].tolist(),
            'metrics': {
                'train_rmse': t_rmse,
                'val_rmse':   v_rmse,
                'rmse_diff':  rmse_diff,
                'val_r2':     v_r2,
            },
            'equations': raw_eqs,
            'plot_data': {
            't':         t,
            'X':         X,
            'train_idx': train_idx,
            'val_idx':   val_idx,
            'x_sim':     x_sim_full,
            },
        }
        render_plot(counter[0])
        # # Plot — scatter train/val points, line = SINDy simulation
        # p.renderers  = []
        # # instead of call directly p.legend.items = []
        # if p.legend:
        #     p.legend.items = []

        # # Train points (blue scatter)
        # p.scatter(t[train_idx], X[train_idx, 0],
        #           color="#1f77b4", alpha=0.4, size=4,
        #           legend_label="Train points")

        # # Validation points (orange scatter)
        # p.scatter(t[val_idx], X[val_idx, 0],
        #           color="#ff7f0e", alpha=0.4, size=4,
        #           legend_label="Val points")

        # # SINDy simulation line on all t
        # if x_sim_full is not None:
        #     p.line(t, x_sim_full[:, 0],
        #            color="#2ecc71", line_width=2.5,
        #            legend_label="SINDy found")

        # p.legend.click_policy = "hide"
        # p.legend.location     = "top_right"

        # res_div.text = (
        #     f"<div style='color:#2c3e50;'>"
        #     f"<b>Run #{counter[0]} Equations:</b><br>{formatted_eqs_html}"
        #     f"<br><small style='color:#7f8c8d;'>"
        #     f"Train R²(dX)={t_r2:.4f} | Val R²(dX)={v_r2:.4f}</small></div>"
        # )

    # automatically run suggestion for the default Coupled-Spring system
    initial_path = os.path.join('data', file_select.value)
    if os.path.exists(initial_path):
        try:
            df_init = pd.read_csv(initial_path).astype(np.float64)
            apply_suggestion(df_init, f"✅ Loaded default pre-set system: <b>{file_select.value}</b>")
        except Exception:
            pass 
    btn_train.on_click(on_train_click)

    # -------------------------------------------------------------------------
    # 5. Layout
    # -------------------------------------------------------------------------
    top_row = row(
        column(file_select,file_input, upload_status, train_s, split_div,library_select,
               poly_s, thr_s, btn_train, width=320),
        column(p,view_div)
    )

    return column(
        top_row,
        Div(text="<hr><b>HISTORY LEADERBOARD — Metrics on dx/dt (derivative space)</b>"),
        row(history_table),
    )