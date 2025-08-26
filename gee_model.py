# train_multi_zip_lstm_chicago_point.py
# Pooled multi-ZIP trainer using a SINGLE Chicago coordinate (no polygons, no per-ZIP geometry),
# MODIS day+night LST (default), naïve baselines, and optional Huber loss.

import os, glob, math
import numpy as np
import pandas as pd

import ee
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# >>> SHAP (added) <<<
import shap
import matplotlib.pyplot as plt
# <<< SHAP (added) >>>


# -------------------- CONFIG --------------------
FOLDER = "zip_code_influenza_incidents"   # <--- change to your folder
OUT_CSV = "predictions_panel_chicago_point.csv"
WINDOW = 6                              # t-5…t -> t+1
SEED = 42
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")
LST_MODE = "daynight"                   # "day", "night", or "daynight"
TRY_HUBER = True
# Fixed Chicago downtown coordinate (lon, lat)
CHI_LON, CHI_LAT = -87.6298, 41.8781
BUFFER_M = 15000.0                      # buffer radius around Chicago point (meters)
# ------------------------------------------------

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# Initialize Earth Engine (authenticate once interactively in your env if needed)
try:
    ee.Initialize(project='lsus-448518')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='lsus-448518')

def monday(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())

def load_folder_panel(folder):
    """Load all CSVs (schema: Week_Start, ZIP_Code, ILI_Activity_Level[, ZIP_Code_Location])."""
    frames = []
    for fp in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(fp)
        need = {"Week_Start","ZIP_Code","ILI_Activity_Level"}
        if not need.issubset(df.columns):
            continue
        df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
        df = df.dropna(subset=["Week_Start"]).copy()
        df["week_start_date"] = df["Week_Start"].apply(monday)
        df["ZIP_Code"] = df["ZIP_Code"].astype(str).str.zfill(5)
        if "ZIP_Code_Location" not in df.columns:
            df["ZIP_Code_Location"] = None
        frames.append(df[["week_start_date","ZIP_Code","ILI_Activity_Level","ZIP_Code_Location"]])
    if not frames:
        raise RuntimeError("No valid CSVs found in folder.")
    panel = pd.concat(frames, ignore_index=True)
    return panel

def collapse_duplicates(panel):
    """Collapse duplicate rows per ZIP×week (mean of ILI_Activity_Level)."""
    agg = (panel
           .groupby(["ZIP_Code","week_start_date"], as_index=False)
           .agg({"ILI_Activity_Level":"mean", "ZIP_Code_Location":"first"}))
    return agg

def chicago_region(buffer_m: float) -> ee.Geometry:
    """Single Chicago point with buffer. No polygons, no per-ZIP geometry."""
    return ee.Geometry.Point([CHI_LON, CHI_LAT]).buffer(buffer_m)

def fetch_weekly_lst(weeks, geom: ee.Geometry, mode="daynight"):
    """
    Weekly mean MODIS LST for the Chicago buffered point region.
    Returns: df with [week_start_date, lst_day_c, lst_night_c, lst_daynight_c]
    """
    col = ee.ImageCollection("MODIS/061/MOD11A1").select(["LST_Day_1km","LST_Night_1km"])
    rows = []
    for wk in weeks:
        start = ee.Date(pd.Timestamp(wk).strftime("%Y-%m-%d"))
        end   = start.advance(7, "day")
        img   = col.filterDate(start, end).mean()
        stats = img.reduceRegion(ee.Reducer.mean(), geom, scale=1000, maxPixels=1e13)
        day   = stats.get("LST_Day_1km").getInfo()
        night = stats.get("LST_Night_1km").getInfo()
        k2c = lambda v: None if v is None else v*0.02 - 273.15
        day_c, night_c = k2c(day), k2c(night)
        if day_c is not None and night_c is not None:
            dn = 0.5*(day_c + night_c)
        else:
            dn = day_c if day_c is not None else night_c
        rows.append({"week_start_date": pd.Timestamp(wk),
                     "lst_day_c": day_c, "lst_night_c": night_c, "lst_daynight_c": dn})
    return pd.DataFrame(rows)

def build_panel_with_city_lst(panel, lst_mode="daynight", buffer_m=15000.0):
    """
    Create ZIP×week grid, attach ILI (collapsed), and merge the SAME citywide LST time series to all ZIPs.
    """
    s, e = panel["week_start_date"].min(), panel["week_start_date"].max()
    weeks = pd.date_range(start=monday(s), end=monday(e), freq="W-MON")
    zips = sorted(panel["ZIP_Code"].unique())
    grid = pd.MultiIndex.from_product([zips, weeks], names=["ZIP_Code","week_start_date"]).to_frame(index=False)
    panel = grid.merge(panel, on=["ZIP_Code","week_start_date"], how="left")

    # One citywide LST series
    geom = chicago_region(buffer_m)
    lst_df = fetch_weekly_lst(weeks, geom, mode=lst_mode)

    # Merge city LST to all ZIP rows by week
    data = panel.merge(lst_df, on="week_start_date", how="left")
    # Choose LST feature
    if lst_mode == "day":
        data["LST_FEATURE"] = data["lst_day_c"]
    elif lst_mode == "night":
        data["LST_FEATURE"] = data["lst_night_c"]
    else:
        data["LST_FEATURE"] = data["lst_daynight_c"]

    data = data.dropna(subset=["ILI_Activity_Level","LST_FEATURE"]).sort_values(["ZIP_Code","week_start_date"])
    return data

def make_sequences_by_zip(df, X_scaled, y_scaled, window):
    """
    Build sequences ZIP-by-ZIP; also return metadata mapping (ZIP, target_week).
    """
    Xs, ys, meta = [], [], []
    start = 0
    for z, g in df.groupby("ZIP_Code", sort=True):
        n = len(g)
        Xg, yg = X_scaled[start:start+n, :], y_scaled[start:start+n]
        weeks = g["week_start_date"].tolist()
        start += n
        for i in range(n - window):
            Xs.append(Xg[i:i+window, :])
            ys.append(yg[i+window])
            meta.append((z, weeks[i+window]))  # target week
    return np.asarray(Xs), np.asarray(ys), meta

def train_eval_model(X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, loss="mse"):
    tf.keras.backend.clear_session()
    loss_fn = tf.keras.losses.Huber(delta=1.0) if loss=="huber" else "mse"
    model = models.Sequential([
        layers.Input(shape=(WINDOW, 2)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_trn_seq, y_trn_seq,
              validation_data=(X_val_seq, y_val_seq),
              epochs=100, batch_size=32, callbacks=[es], verbose=1)
    # Predict & invert scale
    y_pred_scaled = model.predict(X_tst_seq).reshape(-1,1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_tst_seq.reshape(-1,1)).ravel()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return model, (y_true, y_pred, rmse, mae)

def eval_baselines_from_meta(meta, actual_series):
    """
    Compute Naive-1 and Seasonal-naive using the (ZIP, target_week) meta mapping.
    actual_series is a Series indexed by (ZIP_Code, week_start_date) -> ILI value.
    """
    naive_true, naive_pred = [], []
    seas_true, seas_pred   = [], []
    for (z, wk) in meta:
        y_t  = actual_series.get((z, wk), np.nan)
        y_t1 = actual_series.get((z, wk - pd.Timedelta(days=7)), np.nan)
        y_t52= actual_series.get((z, wk - pd.Timedelta(days=364)), np.nan)
        if not (np.isnan(y_t) or np.isnan(y_t1)):
            naive_true.append(y_t); naive_pred.append(y_t1)
        if not (np.isnan(y_t) or np.isnan(y_t52)):
            seas_true.append(y_t); seas_pred.append(y_t52)
    def score(y_true, y_pred):
        if len(y_true)==0: return (None, None)
        return (math.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred))
    return score(naive_true, naive_pred), score(seas_true, seas_pred)

# -------------------- PIPELINE --------------------
print("Loading CSVs...")
panel_raw = load_folder_panel(FOLDER)
print(f"Loaded rows: {len(panel_raw):,}")

print("Collapsing duplicates per ZIP × week ...")
panel = collapse_duplicates(panel_raw)

print("Building panel with CITYWIDE MODIS LST from Chicago point buffer (no polygons, no per-ZIP geometry) ...")
data = build_panel_with_city_lst(panel, lst_mode=LST_MODE, buffer_m=BUFFER_M)

# Split by time for ALL ZIPs (same boundaries)
train_mask = data["week_start_date"] <= TRAIN_END
test_mask  = data["week_start_date"] >= TEST_START
train_df = data.loc[train_mask].copy()
test_df  = data.loc[test_mask].copy()

# Validation = last 20% per ZIP
def split_zip_train_val(g):
    n = len(g)
    v = max(int(0.2*n), WINDOW+5)
    return g.iloc[:n-v], g.iloc[n-v:]

trn_parts, val_parts = [], []
for z, g in train_df.groupby("ZIP_Code", sort=True):
    a, b = split_zip_train_val(g.sort_values("week_start_date"))
    if len(a) > WINDOW and len(b) > 0:
        trn_parts.append(a); val_parts.append(b)

trn_df = pd.concat(trn_parts).sort_values(["ZIP_Code","week_start_date"])
val_df = pd.concat(val_parts).sort_values(["ZIP_Code","week_start_date"])

# Scale on TRAIN only (pooled)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

def pack(df): return df[["ILI_Activity_Level","LST_FEATURE"]].values

X_trn = scaler_X.fit_transform(pack(trn_df))
X_val = scaler_X.transform(pack(val_df))
X_tst = scaler_X.transform(pack(test_df))

y_trn = scaler_y.fit_transform(trn_df[["ILI_Activity_Level"]]).ravel()
y_val = scaler_y.transform(val_df[["ILI_Activity_Level"]]).ravel()
y_tst = scaler_y.transform(test_df[["ILI_Activity_Level"]]).ravel()

# Sequences + meta mapping
X_trn_seq, y_trn_seq, meta_trn = make_sequences_by_zip(trn_df, X_trn, y_trn, WINDOW)
X_val_seq, y_val_seq, meta_val = make_sequences_by_zip(val_df, X_val, y_val, WINDOW)
X_tst_seq, y_tst_seq, meta_tst = make_sequences_by_zip(test_df, X_tst, y_tst, WINDOW)

print("Sequences -> Train:", X_trn_seq.shape, "Val:", X_val_seq.shape, "Test:", X_tst_seq.shape)

# Train/Eval (MSE)
model_mse, (y_true, y_pred_mse, rmse_mse, mae_mse) = train_eval_model(
    X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, loss="mse"
)
print(f"[MSE]   TEST RMSE: {rmse_mse:.3f} | TEST MAE: {mae_mse:.3f}")

# Optional Train/Eval (Huber)
y_pred_huber = None; rmse_huber = mae_huber = None
if TRY_HUBER:
    model_huber, (y_true_h, y_pred_huber, rmse_huber, mae_huber) = train_eval_model(
        X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, loss="huber"
    )
    print(f"[Huber] TEST RMSE: {rmse_huber:.3f} | TEST MAE: {mae_huber:.3f}")

# Baselines using per-ZIP meta mapping
actual_series = data.set_index(["ZIP_Code","week_start_date"])["ILI_Activity_Level"]
(na_rmse, na_mae), (sz_rmse, sz_mae) = eval_baselines_from_meta(meta_tst, actual_series)
print(f"[Naive-1]        RMSE: {na_rmse if na_rmse else 'NA'} | MAE: {na_mae if na_mae else 'NA'}")
print(f"[Seasonal-Naive] RMSE: {sz_rmse if sz_rmse else 'NA'} | MAE: {sz_mae if sz_mae else 'NA'}")

# Per-ZIP, per-week output aligned via meta
out = pd.DataFrame(meta_tst, columns=["ZIP_Code","week_start_date"])
out["ILI_true"] = y_true
out["ILI_pred_mse"] = y_pred_mse
if y_pred_huber is not None:
    out["ILI_pred_huber"] = y_pred_huber
# Attach LST feature used for that target week
out = out.merge(data[["ZIP_Code","week_start_date","LST_FEATURE"]].drop_duplicates(),
                on=["ZIP_Code","week_start_date"], how="left")
out.to_csv(OUT_CSV, index=False)
print("Saved predictions to:", OUT_CSV)

# ---------------- SHAP (added): lag-wise contribution of LST ----------------
try:
    # Choose best available trained model (Huber if available)
    explained_model = model_huber if (TRY_HUBER and (y_pred_huber is not None)) else model_mse

    # Background sample from TRAIN sequences
    rng = np.random.default_rng(SEED)
    bg_n = min(256, X_trn_seq.shape[0])
    bg_idx = rng.choice(X_trn_seq.shape[0], size=bg_n, replace=False)
    background = X_trn_seq[bg_idx]  # (bg_n, WINDOW, n_features)

    # Subset of TEST sequences to explain
    test_n = min(1024, X_tst_seq.shape[0])
    test_idx = rng.choice(X_tst_seq.shape[0], size=test_n, replace=False)
    X_tst_sub = X_tst_seq[test_idx]  # (test_n, WINDOW, n_features)

    n_features = X_trn_seq.shape[2]
    LST_IDX = 1  # pack(): ["ILI_Activity_Level","LST_FEATURE"]

    shap_arr = None
    # ---- Preferred: GradientExplainer (TF2 eager) ----
    try:
        explainer = shap.GradientExplainer(explained_model, background)
        sv = explainer.shap_values(X_tst_sub)
        shap_arr = sv[0] if isinstance(sv, list) else sv  # (N, WINDOW, n_features)
    except Exception as e_grad:
        print("GradientExplainer failed, falling back to KernelExplainer:", repr(e_grad))
        # ---- Fallback: KernelExplainer (model-agnostic) ----
        def f(X2d):
            X3d = X2d.reshape((-1, WINDOW, n_features))
            return explained_model.predict(X3d, verbose=0)

        bg2d   = background.reshape(background.shape[0], -1)
        test2d = X_tst_sub.reshape(X_tst_sub.shape[0], -1)

        kexp = shap.KernelExplainer(f, bg2d, link="identity")
        shap_vals_2d = kexp.shap_values(test2d, nsamples=200)  # keep modest for speed
        shap_2d = shap_vals_2d[0] if isinstance(shap_vals_2d, list) else shap_vals_2d
        shap_arr = shap_2d.reshape(X_tst_sub.shape[0], WINDOW, n_features)

    # ---- If SHAP succeeded, compute lag-wise |SHAP| for LST ----
    lag_labels = [f"t-{(WINDOW-1 - i)}" for i in range(WINDOW)]  # ["t-5",...,"t-0"]
    if shap_arr is not None:
        lag_importance_shap = np.mean(np.abs(shap_arr[:, :, LST_IDX]), axis=0)  # (WINDOW,)
        # Save numeric summary (ensure 1-D)
        pd.DataFrame({
            "lag": np.array(lag_labels, dtype=object),
            "mean_abs_shap_LST": np.asarray(lag_importance_shap).ravel()
        }).to_csv("shap_lag_importance_LST.csv", index=False)

        # Plot and save bar chart
        plt.figure(figsize=(7, 4))
        plt.bar(lag_labels, np.asarray(lag_importance_shap).ravel())
        plt.title("Lag-wise contribution of LST (mean |SHAP|)")
        plt.xlabel("Lag within look-back window")
        plt.ylabel("Mean |SHAP| for LST")
        plt.tight_layout()
        plt.savefig("shap_lag_importance_LST.png", dpi=300)
        plt.close()
        print("Saved SHAP lag-importance figure: shap_lag_importance_LST.png")
    else:
        raise RuntimeError("SHAP array is None after both explainers.")

except Exception as e:
    print("SHAP attribution failed; using occlusion instead:", repr(e))
    # --------- Occlusion fallback (always works) ----------
    try:
        # Base predictions (inverse-transformed to original units for interpretability)
        y_base_scaled = explained_model.predict(X_tst_sub, verbose=0).reshape(-1, 1)
        y_base = scaler_y.inverse_transform(y_base_scaled).ravel()

        # Use the training mean (scaled space) for LST replacement
        lst_train_mean = np.nanmean(X_trn_seq[:, :, LST_IDX])

        lag_labels = [f"t-{(WINDOW-1 - i)}" for i in range(WINDOW)]
        lag_importance_occ = []
        for lag_i in range(WINDOW):
            X_occ = X_tst_sub.copy()
            X_occ[:, lag_i, LST_IDX] = lst_train_mean
            y_occ_scaled = explained_model.predict(X_occ, verbose=0).reshape(-1, 1)
            y_occ = scaler_y.inverse_transform(y_occ_scaled).ravel()
            delta = np.mean(np.abs(y_base - y_occ))
            lag_importance_occ.append(delta)

        # Save numeric summary
        pd.DataFrame({
            "lag": np.array(lag_labels, dtype=object),
            "occlusion_abs_delta_pred": np.asarray(lag_importance_occ).ravel()
        }).to_csv("occlusion_lag_importance_LST.csv", index=False)

        # Plot and save bar chart
        plt.figure(figsize=(7, 4))
        plt.bar(lag_labels, np.asarray(lag_importance_occ).ravel())
        plt.title("Lag-wise contribution of LST (occlusion |Δ prediction|)")
        plt.xlabel("Lag within look-back window")
        plt.ylabel("|Δ prediction| (original units)")
        plt.tight_layout()
        plt.savefig("occlusion_lag_importance_LST.png", dpi=300)
        plt.close()
        print("Saved occlusion lag-importance figure: occlusion_lag_importance_LST.png")
    except Exception as e2:
        print("Occlusion attribution also failed:", repr(e2))
# ---------------------------------------------------------------------------
