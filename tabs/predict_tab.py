# tabs/predict_tab.py

from bokeh.models import ColumnDataSource, Slider, Button, Select, Div, TextInput
from bokeh.layouts import column, row
from bokeh.plotting import figure
import numpy as np


def predict_tab_layout(engine, trained_model_storage):
    # -------------------------------------------------------------------------
    # 1. UI Components
    # -------------------------------------------------------------------------
    model_select = Select(title="SELECT MODEL (FROM HISTORY)", options=[], value="")
    horizon_s    = Slider(start=10, end=500, value=100, step=10,
                          title="Prediction Horizon (seconds)")
    btn_predict  = Button(label="▶ PREDICT", button_type="primary", height=50)

    # IC input — show the hint based on feature names of selected model 
    ic_hint_div = Div(
        text="<i style='color:#888;font-size:12px;'>Select a model to see variable names.</i>",
        styles={'padding': '2px 0'}
    )
    
    # Sample initial condition
    ic_input = TextInput(
        title="Initial Conditions x₀",
        placeholder="e.g. 1.0, 0.0, 0.5, 0.0",
        value="",
        width=300,
    )

    # -------------------------------------------------------------------------
    # 2. Update IC hint when choosing a different model
    # -------------------------------------------------------------------------
    def on_model_select_change(attr, old, new):
        if not new:
            return
        try:
            run_id = int(new.replace("Run #", ""))
        except ValueError:
            return
        if run_id not in trained_model_storage:
            return

        saved_data = trained_model_storage[run_id]
        names      = saved_data.get('feature_names', [])
        n_vars     = len(names)

        ic_list = saved_data.get('initial_conditions', [1.0] + [0.0] * (n_vars - 1))

        if names:
            # convert list to string
            ic_input.value = ", ".join([f"{v:.4f}" for v in ic_list])
            
            ic_hint_div.text = (
                f"<i style='color:#2980b9;font-size:12px;'>"
                f"Variables: <b>{', '.join(names)}</b> "
                f"— enter {n_vars} values</i>"
            )
        else:
            ic_hint_div.text = "<i style='color:red;'>⚠ Feature names missing.</i>"

    model_select.on_change('value', on_model_select_change)

    # -------------------------------------------------------------------------
    # 3. Plot
    # -------------------------------------------------------------------------
    p_pred = figure(title="Future Trajectory Prediction",
                    sizing_mode="stretch_width", height=500,
                    x_axis_label="Time (s)", y_axis_label="State")
    p_pred.scatter([], [], alpha=0)
    p_pred.legend.click_policy = "hide"

    source_pred = ColumnDataSource(data={})
    renderers   = {}
    colors      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b"]

    status_div = Div(text="", styles={'padding': '4px 0', 'font-size': '13px'})

    # -------------------------------------------------------------------------
    # 4. Predict callback
    # -------------------------------------------------------------------------
    def on_predict_click():
        if not model_select.value:
            status_div.text = "<span style='color:red;'>⚠ No model selected.</span>"
            return

        try:
            run_id = int(model_select.value.replace("Run #", ""))
        except ValueError:
            return

        if run_id not in trained_model_storage:
            return

        saved_data     = trained_model_storage[run_id]
        model_instance = saved_data['model_instance']
        names          = saved_data.get('feature_names', [])
        n_vars         = len(names) if names else model_instance.n_features_in_

        # Parse IC from TextInput
        ic_str = ic_input.value.strip()
        if ic_str:
            try:
                x0 = [float(v.strip()) for v in ic_str.split(',')]
                if len(x0) != n_vars:
                    status_div.text = (
                        f"<span style='color:red;'>⚠ Expected {n_vars} values, "
                        f"got {len(x0)}. Using default IC.</span>"
                    )
                    x0 = [1.0] + [0.0] * (n_vars - 1)
            except ValueError:
                status_div.text = (
                    "<span style='color:red;'>⚠ Invalid IC format. Using default.</span>"
                )
                x0 = [1.0] + [0.0] * (n_vars - 1)
        else:
            x0 = [1.0] + [0.0] * (n_vars - 1)

        t_future = np.linspace(0, horizon_s.value, 1000)

        x_future = engine.simulate_with_model(model_instance, x0, t_future)

        # Clear plot
        p_pred.renderers    = []
        p_pred.legend.items = []
        renderers.clear()

        # Update data
        var_names = names if names else [f"x{i+1}" for i in range(n_vars)]
        data_dict = {'t': t_future}
        for i, name in enumerate(var_names):
            data_dict[name] = x_future[:, i]
        source_pred.data = data_dict

        # Plot lines
        for i, name in enumerate(var_names):
            renderers[name] = p_pred.line(
                't', name,
                source=source_pred,
                color=colors[i % len(colors)],
                line_width=2,
                legend_label=f"Predicted {name}"
            )

        p_pred.legend.click_policy = "hide"
        p_pred.legend.location     = "top_right"
        p_pred.title.text = (
            f"Future Trajectory — Run #{run_id} | "
            f"x₀ = [{', '.join([f'{v:.2g}' for v in x0])}]"
        )

        ic_display = ", ".join([f"{name}={v:.2g}" for name, v in zip(var_names, x0)])
        status_div.text = (
            f"<span style='color:#27ae60;'>✅ Predicted {horizon_s.value}s "
            f"from x₀: {ic_display}</span>"
        )

    btn_predict.on_click(on_predict_click)

    # -------------------------------------------------------------------------
    # 5. Update model list (called on tab switch from main.py)
    # -------------------------------------------------------------------------
    def update_model_list():
        options = [f"Run #{id}" for id in sorted(trained_model_storage.keys())]
        model_select.options = options
        if options:
            if model_select.value not in options:
                model_select.value = options[-1]
                # trigger IC hint update
                on_model_select_change(None, None, model_select.value)

    # -------------------------------------------------------------------------
    # 6. Layout
    # -------------------------------------------------------------------------
    layout = column(
        row(model_select, horizon_s, btn_predict),
        row(
            column(
                ic_hint_div,
                ic_input,
                width=350
            ),
            column(status_div)
        ),
        p_pred,
        sizing_mode="stretch_width"
    )

    return layout, update_model_list