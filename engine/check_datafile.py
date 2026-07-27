# engine/check_datafile.py
#
# Shared CSV upload guards used by both Train and Test tabs, so a
# malformed or oversized upload fails fast with a clear message instead
# of propagating into pySINDy/solve_ivp and crashing with a low-level error.

import numpy as np

MAX_UPLOAD_BYTES = 5_000_000  # ~5MB of base64 text


def check_upload_size(b64_data):
    """Return an error string if the base64 payload is too large, else None."""
    if b64_data and len(b64_data) > MAX_UPLOAD_BYTES:
        return "File too large (max 5MB)."
    return None


def validate_dataframe(df):
    """Sanity-check a loaded CSV DataFrame. Returns an error string or None."""
    if df.shape[1] < 2:
        return "CSV must have at least 2 columns: time + one state variable."
    if df.isnull().values.any():
        return "CSV contains missing/NaN values — please clean the data first."
    if not np.isfinite(df.values).all():
        return "CSV contains infinite values — please check the data."
    return None