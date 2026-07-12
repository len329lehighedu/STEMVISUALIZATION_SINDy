# tabs/test_tab.py

from bokeh.models import ColumnDataSource, Button, Select, Div, DataTable, TableColumn, FileInput
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
import os
import base64
import io
from scipy.integrate import solve_ivp

def test_tab_layout(engine, trained_model_storage):
    # -------------------------------------------------------------------------
    # 1. UI Components
    # -------------------------------------------------------------------------
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]

    model_select = Select(title="SELECT MODEL (FROM HISTORY)", options=[], value="")
    file_select_1 = Select(title="Test Run 1 — CSV File", options=csv_files, value="")
    file_select_2 = Select(title="Test Run 2 — CSV File (optional)", options=["(none)"] + csv_files, value="(none)")
    
    # Upload widgets (hidden by default)
    file_input_test1 = FileInput(accept=".csv", title="Upload Test Data 1", visible=False)
    file_input_test2 = FileInput(accept=".csv", title="Upload Test Data 2", visible=False)

    # =========================================================================
    # KEY FIX: Buffer save base64 data in memory, not using filename
    # =========================================================================
    _upload_buffer = {
        'test1': None,  # Save raw base64 string when upload
        'test2': None,
    }

    def get_test_filenames_list(model_run_id): 
        if model_run_id not in trained_model_storage: return ["", "(none)"]
        train_file = trained_model_storage[model_run_id].get('system_name', '')
        if train_file == "cs_train_data.csv": return ["cs_test_data_1.csv", "cs_test_data_2.csv"]
        if train_file == "vanderpol_train.csv": return ["vanderpol_test_1.csv", "vanderpol_test_2.csv"]
        return ["", "(none)"]

    def update_ui_on_model_select(attr, old, new):
        if not new: return
        try:
            run_id = int(new.replace("Run #", ""))
            train_file = trained_model_storage[run_id].get('system_name', '')
            is_custom = "custom" in train_file or train_file == "custom_upload"
            
            # Dynamic Toggle Visibility
            file_select_1.visible = not is_custom
            file_select_2.visible = not is_custom
            file_input_test1.visible = is_custom
            file_input_test2.visible = is_custom
            
            # Reset buffer when change model
            _upload_buffer['test1'] = None
            _upload_buffer['test2'] = None
            status_div1.text = "<i>Select a model to start.</i>"
            status_div2.text = "<i>Select a model to start.</i>"
            
            if not is_custom:
                targets = get_test_filenames_list(run_id)
                file_select_1.value, file_select_2.value = targets[0], targets[1]
        except Exception as e:
            print(f"UI Update Error: {e}")

    model_select.on_change('value', update_ui_on_model_select)
    btn_test = Button(label="TEST", button_type="primary", height=50,width=100)
    
    # -------------------------------------------------------------------------
    # 2. Upload Logic — only cache base64, not reading filename
    # -------------------------------------------------------------------------
    def on_upload_test1(attr, old, new):
        if not new:
            return
        _upload_buffer['test1'] = new          # cache raw base64
        status_div1.text = "<b style='color:green;'>✅ Test file 1 ready. Click RUN TEST.</b>"

    def on_upload_test2(attr, old, new):
        if not new:
            return
        _upload_buffer['test2'] = new          # cache raw base64
        status_div2.text = "<b style='color:green;'>✅ Test file 2 ready. Click RUN TEST.</b>"

    file_input_test1.on_change('value', on_upload_test1)
    file_input_test2.on_change('value', on_upload_test2)

    status_div1 = Div(
        text="<i>Select a model to start.</i>",
        styles={'padding': '8px'}
    )
    status_div2 = Div(
        text="<i>Select a model to start.</i>",
        styles={'padding': '8px'}
    )

    # -------------------------------------------------------------------------
    # 3. Metrics Table & Plots
    # -------------------------------------------------------------------------
    source_metrics = ColumnDataSource(data=dict(
        run=[], model_run=[], variable=[], rmse=[], mae=[], r2=[]
    ))
    metrics_table = DataTable(source=source_metrics, columns=[
        TableColumn(field="run",       title="Test Run", width=100),
        TableColumn(field="model_run", title="Model",    width=100),
        TableColumn(field="variable",  title="Var",      width=100),
        TableColumn(field="rmse",      title="RMSE",     width=120),
        TableColumn(field="r2",        title="R²",       width=120),
    ], sizing_mode="stretch_width", height=300)

    p1 = figure(title="Test Run 1", width=900, height=350, x_axis_label="Time (s)", sizing_mode="stretch_width")
    p2 = figure(title="Test Run 2", width=900, height=350, x_axis_label="Time (s)", visible=False, sizing_mode="stretch_width")

    # -------------------------------------------------------------------------
    # 4. Core test runner — use direct df instead of csv path
    # -------------------------------------------------------------------------
    def _run_single_test(fig, df, label):
        """Run test from DataFrame (no need for file path)."""
        t = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

        def rhs(t_val, x):
            return model_instance.predict(np.array(x).reshape(1, -1)).flatten()

        sol = solve_ivp(rhs, (t[0], t[-1]), X[0, :], t_eval=t, method='RK45')
        if not sol.success:
            return None, sol.message

        # Refresh plot
        fig.renderers = []
        if fig.legend:
            fig.legend.items = []

        colors = Category10[10]
        rows = []
        for i, vname in enumerate(df.columns[1:]):
            fig.scatter(t, X[:, i], color=colors[i * 2],     alpha=0.3, legend_label=f"{vname} (True)")
            fig.line(   t, sol.y[i], color=colors[i * 2 + 1], line_width=2, legend_label=f"{vname} (SINDy)")

            res = X[:, i] - sol.y[i]
            r2  = 1 - (np.sum(res**2) / np.sum((X[:, i] - np.mean(X[:, i]))**2))
            rows.append({
                'variable': vname,
                'rmse':     f"{np.sqrt(np.mean(res**2)):.6f}",
                'r2':       f"{r2:.4f}",
            })

        fig.legend.click_policy = "hide"
        fig.legend.location     = "top_right"
        return rows, None

    def _load_df_from_select(sel_value):
        """Load DataFrame from file on local machine (preset CSV)."""
        path = os.path.join('data', sel_value)
        if not os.path.exists(path):
            return None, f"File {sel_value} missing"
        return pd.read_csv(path).astype(np.float64), None

    def _load_df_from_buffer(b64_data):
        """Load DataFrame from base64 buffer (uploaded file)."""
        try:
            decoded = base64.b64decode(b64_data)
            df = pd.read_csv(io.BytesIO(decoded)).astype(np.float64)
            return df, None
        except Exception as e:
            return None, str(e)

    # -------------------------------------------------------------------------
    # 5. RUN TEST callback
    # -------------------------------------------------------------------------
    def on_test_click():
        if not model_select.value:
            return

        run_id = int(model_select.value.replace("Run #", ""))
        model_data = trained_model_storage[run_id]

        # Use nonlocal model_instance so _run_single_test can get access to
        nonlocal model_instance
        model_instance = model_data['model_instance']

        run_id_str = f"#{run_id}"
        res_data   = {'run': [], 'model_run': [], 'variable': [], 'rmse': [], 'r2': []}

        # --- Test Run 1 ---
        is_custom = file_input_test1.visible
        if is_custom:
            if not _upload_buffer['test1']:
                status_div1.text = "<b style='color:red;'>⚠️ Please upload Test file 1 first.</b>"
                return
            df, err = _load_df_from_buffer(_upload_buffer['test1'])
        else:
            df, err = _load_df_from_select(file_select_1.value)

        if err:
            status_div1.text = f"<b style='color:red;'>⚠️ Run 1 error: {err}</b>"
        elif df is not None:
            rows, err = _run_single_test(p1, df, "Run 1")
            if rows:
                p1.visible = True
                for r in rows:
                    res_data['variable'].append(r['variable'])
                    res_data['rmse'].append(r['rmse'])
                    res_data['r2'].append(r['r2'])
                    res_data['run'].append("Run 1")
                    res_data['model_run'].append(run_id_str)

        # --- Test Run 2 ---
        if is_custom:
            if _upload_buffer['test2']:
                df2, err2 = _load_df_from_buffer(_upload_buffer['test2'])
            else:
                df2, err2 = None, None   # optional
        else:
            df2, err2 = _load_df_from_select(file_select_2.value) \
                if file_select_2.value != "(none)" else (None, None)

        if err2:
            status_div2.text = f"<b style='color:red;'>⚠️ Run 2 error: {err2}</b>"
        elif df2 is not None:
            rows2, err2 = _run_single_test(p2, df2, "Run 2")
            if rows2:
                p2.visible = True
                for r in rows2:
                    res_data['variable'].append(r['variable'])
                    res_data['rmse'].append(r['rmse'])
                    res_data['r2'].append(r['r2'])
                    res_data['run'].append("Run 2")
                    res_data['model_run'].append(run_id_str)

        source_metrics.data = res_data
        if not err and not err2:
            status_div1.text = "<b style='color:247008;'>✅ Test complete!</b>"
            status_div2.text = "<b style='color:247008;'>✅ Test complete!</b>"

    # Placeholder for _run_single_test
    model_instance = None
    btn_test.on_click(on_test_click)

    # -------------------------------------------------------------------------
    # 6. update_model_list (called from main app)
    # -------------------------------------------------------------------------
    def update_model_list():
        opts = [f"Run #{i}" for i in sorted(trained_model_storage.keys())]
        model_select.options = opts
        if opts and not model_select.value:
            model_select.value = opts[-1]
        f_list = [f for f in os.listdir('data') if f.endswith('.csv')]
        file_select_1.options = f_list
        file_select_2.options = ["(none)"] + f_list

    # -------------------------------------------------------------------------
    # 7. Layout
    # -------------------------------------------------------------------------
    layout = column(
        Div(text="<h3>🧪 Test Evaluation</h3>"),
        row(
            column(model_select,btn_test),
            column(file_select_1, file_input_test1,status_div1),
            column(file_select_2, file_input_test2,status_div2),
        ),
        metrics_table, p1, p2,
        sizing_mode="stretch_width"
    )
    return layout, update_model_list