# train_multi_zip_lstm_chicago_point.py
# Pooled multi-ZIP trainer using a SINGLE Chicago coordinate (no polygons),
# citywide MODIS day+night LST + Absolute Humidity (ERA5-Land) + PM2.5 (CAMS if available),
# robust feature gating, flexible time split, baselines, early stopping, and optional Huber loss.

import os, glob, math, re
import numpy as np
import pandas as pd

import ee
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------- CONFIG --------------------
FOLDER = "zip_code_influenza_incidents"   # <--- set to your folder with ZIP CSVs
OUT_CSV = "predictions_panel_chicago_point.csv"
WINDOW = 6                                # t-5…t -> t+1
SEED = 42

# Preferred boundary; script falls back to global 80/20 if this yields too little train data
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

LST_MODE = "daynight"                     # "day", "night", or "daynight"
TRY_HUBER = True

# Fixed Chicago downtown coordinate (lon, lat)
CHI_LON, CHI_LAT = -87.6298, 41.8781
BUFFER_M = 15000.0                        # buffer radius around Chicago point (meters)
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

# -------------------- LOAD & PREP --------------------

def load_folder_panel(folder):
    """Load all CSVs: require Week_Start, ZIP_Code, ILI_Activity_Level; keep optional ZIP_Code_Location."""
    frames = []
    for fp in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(fp)
        need = {"Week_Start","ZIP_Code","ILI_Activity_Level"}
        if not need.issubset(df.columns):
            continue
        df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
        df = df.dropna(subset=["Week_Start"]).copy()
        df["week_start_date"] = df["Week_Start"].apply(monday)  # ensure Monday-aligned
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

# -------------------- ENV FEATURES --------------------

def fetch_weekly_lst(weeks_list, geom: ee.Geometry, mode="daynight"):
    """Weekly mean MODIS LST (°C) for the Chicago buffered point."""
    col = ee.ImageCollection("MODIS/061/MOD11A1").select(["LST_Day_1km","LST_Night_1km"])
    rows = []
    for wk in weeks_list:
        start = ee.Date(pd.Timestamp(wk).strftime("%Y-%m-%d"))
        end   = start.advance(7, "day")
        img   = col.filterDate(start, end).mean()
        stats = img.reduceRegion(ee.Reducer.mean(), geom, scale=1000, maxPixels=1e13, bestEffort=True)
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

def fetch_weekly_abs_humidity(weeks_list, geom: ee.Geometry):
    """Weekly mean Absolute Humidity (g/m^3) using ERA5-Land hourly T2m & D2m."""
    col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select(
        ["temperature_2m","dewpoint_temperature_2m"]
    )
    rows = []
    for wk in weeks_list:
        start = ee.Date(pd.Timestamp(wk).strftime("%Y-%m-%d"))
        end   = start.advance(7, "day")
        img   = col.filterDate(start, end).mean()
        stats = img.reduceRegion(ee.Reducer.mean(), geom, scale=10000, maxPixels=1e13, bestEffort=True)
        tK = stats.get("temperature_2m").getInfo()          # Kelvin
        dK = stats.get("dewpoint_temperature_2m").getInfo() # Kelvin
        if tK is None or dK is None:
            ah = None
        else:
            T  = tK - 273.15
            Td = dK - 273.15
            e_hPa = 6.112 * math.exp(17.67*Td/(Td + 243.5))
            ah = (2.1674 * e_hPa) / (273.15 + T)  # g/m^3
        rows.append({"week_start_date": pd.Timestamp(wk), "ah_gm3": ah})
    return pd.DataFrame(rows)

def _pick_pm25_band(band_names):
    """
    Choose a PM2.5 band name robustly across CAMS variants.
    Priority: exact official long name if present; else any band containing pm/2p5/25 + 'surface'.
    """
    exact = "particulate_matter_d_less_than_25_um_surface"  # CAMS PM2.5 surface mass (kg/m^3)
    if exact in band_names:
        return exact
    # Heuristic fallbacks
    patterns = [
        r"particulate_matter.*2p?5.*surface",
        r"pm2\.?5.*surface",
        r"pm2p5.*surface",
        r"pm25.*surface",
    ]
    for pat in patterns:
        for b in band_names:
            if re.search(pat, b.replace(" ", "").lower()):
                return b
    return None

def _resolve_pm25_source(sample_date_iso: str):
    """Probe CAMS datasets/bands to find an available PM2.5 source."""
    candidates = [
        "ECMWF/CAMS/NRT",
        "ECMWF/CAMS/REANALYSIS",
    ]
    for ds in candidates:
        try:
            col = ee.ImageCollection(ds).filterDate(
                ee.Date(sample_date_iso).advance(-7, "day"),
                ee.Date(sample_date_iso).advance(7, "day")
            )
            img = col.first()
            if img is None:
                continue
            names = img.bandNames().getInfo()
            band = _pick_pm25_band(names)
            if band:
                return ds, band
        except Exception:
            continue
    return None, None

def fetch_weekly_pm25(weeks_list, geom: ee.Geometry):
    """
    Weekly mean surface PM2.5 from CAMS (kg/m^3 -> µg/m^3 by ×1e9).
    Returns: df [week_start_date, pm25_ugm3]
    """
    if weeks_list is None or (hasattr(weeks_list, "__len__") and len(weeks_list) == 0):
        return pd.DataFrame(columns=["week_start_date","pm25_ugm3"])

    sample_iso = pd.Timestamp(weeks_list[len(weeks_list)//2]).strftime("%Y-%m-%d")
    ds, band = _resolve_pm25_source(sample_iso)
    if ds is None or band is None:
        print("[WARN] CAMS PM2.5 band not found; PM2.5 will be NaN and auto-disabled.")
        return pd.DataFrame({"week_start_date": weeks_list, "pm25_ugm3": [None]*len(weeks_list)})

    rows = []
    for wk in weeks_list:
        start = ee.Date(pd.Timestamp(wk).strftime("%Y-%m-%d"))
        end   = start.advance(7, "day")
        col = ee.ImageCollection(ds).select([band]).filterDate(start, end)
        img = col.mean()
        try:
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=50000,          # ~50 km consistent with CAMS native grid (~44.5 km)
                maxPixels=1e13,
                bestEffort=True
            )
            val_kgm3 = stats.get(band).getInfo()
        except Exception:
            val_kgm3 = None

        val_ugm3 = (val_kgm3 * 1e9) if (val_kgm3 is not None) else None
        rows.append({"week_start_date": pd.Timestamp(wk), "pm25_ugm3": val_ugm3})
    return pd.DataFrame(rows)

# -------------------- PANEL BUILD --------------------

def build_panel_with_city_env(panel, lst_mode="daynight", buffer_m=15000.0):
    """
    Create ZIP×week grid, attach ILI (collapsed), and merge SAME citywide environmental series
    (LST, AH, PM2.5) to all ZIP rows by week. Auto-disables AH/PM2.5 if entirely missing.
    """
    s, e = panel["week_start_date"].min(), panel["week_start_date"].max()
    weeks_idx = pd.date_range(start=monday(s), end=monday(e), freq="W-MON")
    weeks = weeks_idx.to_list()  # list for iterators

    zips = sorted(panel["ZIP_Code"].unique())
    grid = pd.MultiIndex.from_product([zips, weeks_idx], names=["ZIP_Code","week_start_date"]).to_frame(index=False)
    panel = grid.merge(panel, on=["ZIP_Code","week_start_date"], how="left")

    geom = chicago_region(buffer_m)

    lst_df = fetch_weekly_lst(weeks, geom, mode=lst_mode)
    ah_df  = fetch_weekly_abs_humidity(weeks, geom)
    pm_df  = fetch_weekly_pm25(weeks, geom)

    # Merge city series onto ZIP×week
    data = panel.merge(lst_df, on="week_start_date", how="left")
    data = data.merge(ah_df,  on="week_start_date", how="left")
    data = data.merge(pm_df,  on="week_start_date", how="left")

    # Choose LST feature
    if lst_mode == "day":
        data["LST_FEATURE"] = data["lst_day_c"]
    elif lst_mode == "night":
        data["LST_FEATURE"] = data["lst_night_c"]
    else:
        data["LST_FEATURE"] = data["lst_daynight_c"]

    # Interpolate within each column; keeps ends via both-direction fill
    for c in ["LST_FEATURE", "ah_gm3", "pm25_ugm3"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].interpolate(limit_direction="both")

    # Decide which env features are usable
    use_ah   = not data["ah_gm3"].isna().all()
    use_pm25 = not data["pm25_ugm3"].isna().all()
    if not use_ah:
        print("[WARN] Absolute humidity is entirely missing after merge; AH feature will be disabled.")
    if not use_pm25:
        print("[WARN] PM2.5 is entirely missing after merge; PM2.5 feature will be disabled.")

    # Keep rows with the required columns
    required = ["ILI_Activity_Level","LST_FEATURE"]
    if use_ah:   required.append("ah_gm3")
    if use_pm25: required.append("pm25_ugm3")

    data = data.dropna(subset=required).sort_values(["ZIP_Code","week_start_date"]).copy()
    return data, use_ah, use_pm25

# -------------------- MODELING --------------------

def make_sequences_by_zip(df, X_scaled, y_scaled, window):
    """Build sequences ZIP-by-ZIP; also return metadata mapping (ZIP, target_week)."""
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

def train_eval_model(X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, n_features, loss="mse"):
    tf.keras.backend.clear_session()
    loss_fn = tf.keras.losses.Huber(delta=1.0) if loss=="huber" else "mse"
    model = models.Sequential([
        layers.Input(shape=(WINDOW, n_features)),
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
    """Compute Naive-1 and Seasonal-naive using the (ZIP, target_week) meta mapping."""
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
print("Panel coverage:",
      f"ZIPs={panel['ZIP_Code'].nunique()}",
      f"weeks={panel['week_start_date'].nunique()}",
      f"date span={panel['week_start_date'].min().date()}→{panel['week_start_date'].max().date()}")

print("Building panel with CITYWIDE env features (LST, AH, PM2.5) from Chicago point buffer ...")
data, use_ah, use_pm25 = build_panel_with_city_env(panel, lst_mode=LST_MODE, buffer_m=BUFFER_M)
print("After env merge/dropna:",
      f"rows={len(data):,}, ZIPs={data['ZIP_Code'].nunique()}, weeks={data['week_start_date'].nunique()}",
      f"use_AH={use_ah}, use_PM25={use_pm25}")

# --------- Flexible train/test split ----------
pre_boundary  = data.loc[data["week_start_date"] <= TRAIN_END]
post_boundary = data.loc[data["week_start_date"] >= TEST_START]

if len(pre_boundary) >= (WINDOW + 2) and len(post_boundary) > 0:
    # Use preferred boundary
    train_df = pre_boundary.copy()
    test_df  = post_boundary.copy()
    split_mode = "boundary_2021/2022"
else:
    # Fall back to global chronological 80/20 split
    data_sorted = data.sort_values("week_start_date")
    split_idx = int(0.8 * len(data_sorted))
    if split_idx <= WINDOW + 1:
        raise RuntimeError(
            f"Not enough usable rows after merges (n={len(data_sorted)}). "
            "Try reducing WINDOW (e.g., 4), increasing BUFFER_M, or disabling scarce features."
        )
    train_df = data_sorted.iloc[:split_idx].copy()
    test_df  = data_sorted.iloc[split_idx:].copy()
    split_mode = "global_80_20"

print(f"Split mode: {split_mode} | train_rows={len(train_df):,} | test_rows={len(test_df):,}")

# ---- Per-ZIP validation with fallback ----
def split_zip_train_val(g, window):
    g = g.sort_values("week_start_date")
    n = len(g)
    if n <= window + 1:
        return None, None
    v = max(int(0.2 * n), window + 5)
    v = min(v, max(1, n // 3))
    a = g.iloc[: n - v]
    b = g.iloc[n - v :]
    if len(a) <= window or len(b) == 0:
        return None, None
    return a, b

trn_parts, val_parts, skipped = [], [], []
for z, g in train_df.groupby("ZIP_Code", sort=True):
    a, b = split_zip_train_val(g, WINDOW)
    if a is None or b is None:
        skipped.append((z, len(g)))
    else:
        trn_parts.append(a); val_parts.append(b)

if len(trn_parts) == 0:
    # Final fallback: split the pooled train_df globally 80/20
    all_train = train_df.sort_values("week_start_date").copy()
    n = len(all_train)
    v = max(int(0.2 * n), WINDOW + 5)
    v = min(v, max(1, n // 3))
    trn_df = all_train.iloc[: n - v].copy()
    val_df = all_train.iloc[n - v :].copy()
    print("[WARN] Per-ZIP validation not possible; using pooled 80/20 validation within train.")
else:
    trn_df = pd.concat(trn_parts).sort_values(["ZIP_Code","week_start_date"]).copy()
    val_df = pd.concat(val_parts).sort_values(["ZIP_Code","week_start_date"]).copy()

print(f"Training ZIPs used: {len(trn_parts)} | Skipped ZIPs (too short): {len(skipped)}")
if skipped:
    print("Example skipped ZIPs (zip, rows):", skipped[:5])

# ------------- Scaling & Sequences -------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Build feature list dynamically based on availability
feature_cols = ["ILI_Activity_Level", "LST_FEATURE"]
if use_ah:   feature_cols.append("ah_gm3")
if use_pm25: feature_cols.append("pm25_ugm3")

def pack(df):
    return df[feature_cols].values

X_trn = scaler_X.fit_transform(pack(trn_df))
X_val = scaler_X.transform(pack(val_df))
X_tst = scaler_X.transform(pack(test_df))

y_trn = scaler_y.fit_transform(trn_df[["ILI_Activity_Level"]]).ravel()
y_val = scaler_y.transform(val_df[["ILI_Activity_Level"]]).ravel()
y_tst = scaler_y.transform(test_df[["ILI_Activity_Level"]]).ravel()

def make_sequences_by_zip(df, X_scaled, y_scaled, window):
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
            meta.append((z, weeks[i+window]))
    return np.asarray(Xs), np.asarray(ys), meta

X_trn_seq, y_trn_seq, meta_trn = make_sequences_by_zip(trn_df, X_trn, y_trn, WINDOW)
X_val_seq, y_val_seq, meta_val = make_sequences_by_zip(val_df, X_val, y_val, WINDOW)
X_tst_seq, y_tst_seq, meta_tst = make_sequences_by_zip(test_df, X_tst, y_tst, WINDOW)

print("Sequences -> Train:", X_trn_seq.shape, "Val:", X_val_seq.shape, "Test:", X_tst_seq.shape)

if X_trn_seq.size == 0 or X_val_seq.size == 0 or X_tst_seq.size == 0:
    raise RuntimeError(
        "Empty sequence arrays. Likely causes: too few weeks after merges/dropna, "
        "WINDOW too large, or features too scarce. "
        "Try reducing WINDOW (e.g., 4) or disabling scarce features."
    )

n_features = X_trn_seq.shape[2]

def train_eval_model(X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, n_features, loss="mse"):
    tf.keras.backend.clear_session()
    loss_fn = tf.keras.losses.Huber(delta=1.0) if loss=="huber" else "mse"
    model = models.Sequential([
        layers.Input(shape=(WINDOW, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_trn_seq, y_trn_seq,
              validation_data=(X_val_seq, y_val_seq),
              epochs=100, batch_size=32, callbacks=[es], verbose=1)
    y_pred_scaled = model.predict(X_tst_seq).reshape(-1,1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_tst_seq.reshape(-1,1)).ravel()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return model, (y_true, y_pred, rmse, mae)

# ---- Train/Eval ----
model_mse, (y_true, y_pred_mse, rmse_mse, mae_mse) = train_eval_model(
    X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, n_features=n_features, loss="mse"
)
print(f"[MSE]   TEST RMSE: {rmse_mse:.3f} | TEST MAE: {mae_mse:.3f}")

y_pred_huber = None; rmse_huber = mae_huber = None
if TRY_HUBER:
    model_huber, (y_true_h, y_pred_huber, rmse_huber, mae_huber) = train_eval_model(
        X_trn_seq, y_trn_seq, X_val_seq, y_val_seq, X_tst_seq, y_tst_seq, scaler_y, n_features=n_features, loss="huber"
    )
    print(f"[Huber] TEST RMSE: {rmse_huber:.3f} | TEST MAE: {mae_huber:.3f}")

# ---- Baselines ----
actual_series = data.set_index(["ZIP_Code","week_start_date"])["ILI_Activity_Level"]
(b_rmse, b_mae), (s_rmse, s_mae) = eval_baselines_from_meta(meta_tst, actual_series)
print(f"[Naive-1]        RMSE: {b_rmse if b_rmse else 'NA'} | MAE: {b_mae if b_mae else 'NA'}")
print(f"[Seasonal-Naive] RMSE: {s_rmse if s_rmse else 'NA'} | MAE: {s_mae if s_mae else 'NA'}")

# ---- Output ----
out = pd.DataFrame(meta_tst, columns=["ZIP_Code","week_start_date"])
out["ILI_true"] = y_true
out["ILI_pred_mse"] = y_pred_mse
if y_pred_huber is not None:
    out["ILI_pred_huber"] = y_pred_huber

# Attach env features used for that target week (only those enabled)
merge_cols = ["ZIP_Code","week_start_date","LST_FEATURE"]
if use_ah:   merge_cols.append("ah_gm3")
if use_pm25: merge_cols.append("pm25_ugm3")

out = out.merge(data[merge_cols].drop_duplicates(), on=["ZIP_Code","week_start_date"], how="left")
out.to_csv(OUT_CSV, index=False)
print("Saved predictions to:", OUT_CSV)
