# =============================================================================
# tabs/ensemble_tab.py
#
# PURPOSE
# -------
# Experimental tab : runs block-bootstrap ensemble
# fitting on an already-trained model from the Train tab, reporting how
# often each candidate term survives thresholding (inclusion frequency)
# and how stable its coefficient is across resamples.
#
# Deliberately synchronous — this is a local-dev experiment, not a
# deployed feature, so no background-thread/executor machinery here.

# Step-by-Step ConceptOriginal Data:
# You have $m$ original data points ($m$ rows in the CSV).
# Generate a Simulated Dataset: Randomly draw $m$ points—allowing duplicates—from the original $m$ points.
# For instance, if the original data is $[1, 2, 3, 4, 5]$, a single draw might yield $[2, 2, 4, 1, 5]$ (the number 2 appears twice, while 3 vanishes).
# It still contains 5 elements, but its composition has changed.
# Fit the Model: Fit SINDy on this simulated dataset to produce a new set of coefficients
# (which may differ slightly from the original set because the input composition changed).
# Repeat: Repeat steps 2 and 3 many times (e.g., 50–100 times) to obtain 50–100 different sets of coefficients.
# Analyze Dispersion: Look at the spread of those 50–100 sets:Low variation:
# If they are all nearly identical, the coefficients are reliable and stable.
# If they fluctuate wildly—or if certain terms appear and disappear—those coefficients/terms are uncertain
# and heavily influenced by random noise in that specific measurement run.
# This directly aligns with what was mentioned earlier: "It's like I manually ran it 50 times and counted."
# Exactly—except that instead of taking 50 actual new measurements (which is impossible with only one file),
# bootstrap simulates 50 "similar" versions of the original dataset via resampling.
#
# Why Resample by BLOCK Instead of Individual Points?
# The data consists of time series where points like $t = 5.00$ and $t = 5.01$ are nearly identical (a smooth trajectory).
# Point-wise resampling: If you resample individual points independently, you risk accidentally picking many near-identical points.
# The simulated dataset might look varied on paper, but the actual information inside changes very little,
# leading you to underestimate true uncertainty.
# Block resampling: Resampling continuous blocks (Block Bootstrap) preserves the smooth temporal structure within each block
# while shuffling/repeating the blocks themselves. This far better reflects the true uncertainty of trajectory data.
# =============================================================================

from bokeh.models import ColumnDataSource, Button, Select, Div, Slider, DataTable, TableColumn
from bokeh.layouts import column, row
from bokeh.plotting import figure


def ensemble_tab_layout(engine, trained_model_storage):
    """
    Read-only against trained_model_storage: never creates a new run,
    only attaches an 'ensemble' result dict onto an existing run_id.
    """
    # Button select, slider, and data table
    model_select = Select(
        title="SELECT MODEL (FROM HISTORY)", options=[], value="")
    n_bootstrap_s = Slider(start=20, end=100, value=50, step=10,
                           title="Bootstrap Samples")
    btn_run = Button(label="RUN ENSEMBLE", button_type="warning",
                     width=150, height=50, disabled=True)
    progress_div = Div(text="<i>Select a model to start.</i>",
                       styles={'padding': '8px'})

    source_incl = ColumnDataSource(data=dict(
        state=[], term=[], incl_pct=[], coef_mean=[], coef_std=[], n_samples=[]))
    incl_table = DataTable(source=source_incl, columns=[
        TableColumn(field="state",     title="State",        width=80),
        TableColumn(field="term",      title="Term",          width=150),
        TableColumn(field="incl_pct",  title="Inclusion %",   width=100),
        TableColumn(field="coef_mean", title="Coef (mean)",   width=120),
        TableColumn(field="coef_std",  title="Coef (std)",    width=120),
        TableColumn(field="n_samples", title="# Samples",     width=100),
    ], sizing_mode="stretch_width", height=400)

    p_incl = figure(x_range=[], title="Term Inclusion Frequency",
                    sizing_mode="stretch_width", height=300,
                    y_axis_label="% of bootstrap runs", toolbar_location=None)

    def on_model_select_change(attr, old, new):
        btn_run.disabled = not bool(new)
        progress_div.text = "<i>Ready to run ensemble on this model.</i>" if new else "<i>Select a model to start.</i>"

    model_select.on_change('value', on_model_select_change)

    def _print_progress(i, total):
        # print in terminal when run bootstrap
        print(f"[Ensemble] bootstrap {i}/{total}")

    def on_run_click():
        if not model_select.value:
            return
        run_id = int(model_select.value.replace("Run #", ""))
        run_data = trained_model_storage.get(run_id)
        if run_data is None:
            progress_div.text = "<span style='color:red;'>⚠ Model no longer available.</span>"
            return

        btn_run.disabled = True
        n_boot = n_bootstrap_s.value
        progress_div.text = f"<i>Running {n_boot} bootstrap samples…</i>"

        X, t = run_data['plot_data']['X'], run_data['plot_data']['t']
        names = run_data['feature_names']
        lib_type = run_data['lib_type']
        poly_degree = run_data['poly_degree']
        threshold = run_data['threshold']

        try:
            result = engine.fit_ensemble(
                X, t, poly_degree, threshold, names,
                lib_type=lib_type, n_bootstrap=n_boot,
                random_seed=run_id * 13, progress_callback=_print_progress)
            trained_model_storage[run_id]['ensemble'] = result
            progress_div.text = "<b style='color:#27ae60;'>✅ Ensemble complete</b>"
            _render_results(result)
        except Exception as e:
            progress_div.text = f"<span style='color:red;'>⚠ Ensemble error: {e}</span>"
        finally:
            btn_run.disabled = False

    def _render_results(result):
        rows = dict(state=[], term=[], incl_pct=[],
                    coef_mean=[], coef_std=[], n_samples=[])
        bar_labels, bar_vals = [], []
        for state_name, stats in result['per_state'].items():
            for term in result['feature_names']:
                pct = stats['inclusion_pct'][term]
                rows['state'].append(state_name)
                rows['term'].append(term)
                rows['incl_pct'].append(f"{pct*100:.1f}%")
                mean = stats['coef_mean'][term]
                std = stats['coef_std'][term]
                rows['coef_mean'].append(
                    f"{mean:.4f}" if mean is not None else "—")
                rows['coef_std'].append(
                    f"{std:.4f}" if std is not None else "—")
                rows['n_samples'].append(stats['n_samples'][term])
                bar_labels.append(f"{state_name}: {term}")
                bar_vals.append(pct * 100)
        source_incl.data = rows

        p_incl.renderers = []
        p_incl.x_range.factors = bar_labels
        p_incl.vbar(x=bar_labels, top=bar_vals, width=0.7, color="#3498db")
        p_incl.xaxis.major_label_orientation = 1.0

    btn_run.on_click(on_run_click)

    def update_model_list():
        """Get the runs from train_tab through trained_model_storage (shared for all tabs)"""
        opts = [f"Run #{i}" for i in sorted(trained_model_storage.keys())]
        model_select.options = opts

    layout = column(
        Div(text="<h3>🎲 Ensemble Analysis (Experimental)</h3>"
                 "<p style='font-size:13px;color:#7f8c8d;'>Block-bootstrap resampling to check "
                 "how stable each discovered term is. Compute-heavy — runs synchronously "
                 "(UI will freeze briefly while running).</p>"),
        row(model_select, n_bootstrap_s, btn_run),
        progress_div,
        p_incl,
        incl_table,
        sizing_mode="stretch_width"
    )
    return layout, update_model_list
