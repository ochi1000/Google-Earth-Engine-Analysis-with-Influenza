# pip install earthengine-api pandas numpy scikit-learn tensorflow --upgrade
import os, math, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import layers, models, callbacks
import ee

FOLDER = "zip_code_influenza_incidents"   # <-- your folder
WINDOW = 6
SEED = 42
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# --- Earth Engine init (authenticate once interactively in your env) ---
# import ee; ee.Authenticate(); ee.Initialize()
ee.Initialize()

# --- Helpers ---
def zcta_geometry(zip_code: str, wkt_point: str = None, buffer_m: float = 3000):
    # Try TIGER ZCTA polygons
    for asset, field in [("TIGER/2018/ZCTA5", "ZCTA5CE10"),
                         ("TIGER/2018/ZCTA5", "ZCTA5CE20"),
                         ("TIGER/2010/ZCTA5", "ZCTA5CE10")]:
        try:
            fc = ee.FeatureCollection(asset).filter(ee.Filter.eq(field, zip_code))
            f = fc.first()
            if f is not None:
                g = f.geometry()
                _ = g.coordinates().getInfo()  # force server call
                return g
        except Exception:
            pass
    # Fallback: buffer a point from WKT "POINT (lon lat)"
    if wkt_point:
        wkt = wkt_point.strip().replace("POINT", "").replace("(", "").replace(")", "")
        lon, lat = [float(x) for x in wkt.split()]
        return ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    raise RuntimeError(f"No geometry for ZIP {zip_code}")

def weekly_lst_for_zip(weeks, geom):
    col = ee.ImageCollection("MODIS/006/MOD11A1").select("LST_Day_1km")
    out = []
    for wk in weeks:
        start = ee.Date(wk.strftime("%Y-%m-%d")); end = start.advance(7, "day")
        stats = col.filterDate(start, end).mean().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=1000, maxPixels=1e13
        )
        val = stats.get("LST_Day_1km").getInfo()
        lst_c = None if val is None else val*0.02 - 273.15
        out.append({"week_start_date": wk, "lst_day_c": lst_c})
    return pd.DataFrame(out)

def monday(ts): return ts - pd.Timedelta(days=ts.weekday())

# --- Load all ZIP CSVs and build a panel ---
panel = []
all_weeks = None
for fp in glob.glob(os.path.join(FOLDER, "*.csv")):
    df = pd.read_csv(fp)
    if not {"Week_Start","ILI_Activity_Level","ZIP_Code"}.issubset(df.columns):
        continue
    df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
    df = df.dropna(subset=["Week_Start"]).copy()
    df["week_start_date"] = df["Week_Start"].apply(monday)
    df = df[["week_start_date", "ZIP_Code", "ILI_Activity_Level", "ZIP_Code_Location"]]
    df["ZIP_Code"] = df["ZIP_Code"].astype(str).str.zfill(5)
    panel.append(df)
    if all_weeks is None:
        s, e = df["week_start_date"].min(), df["week_start_date"].max()
    else:
        s = min(s, df["week_start_date"].min())
        e = max(e, df["week_start_date"].max())

panel = pd.concat(panel, ignore_index=True)
# Build a complete grid (ZIP x week) to ensure alignment
weeks = pd.date_range(start=s, end=e, freq="W-MON")
zips = sorted(panel["ZIP_Code"].unique())
grid = pd.MultiIndex.from_product([zips, weeks], names=["ZIP_Code","week_start_date"]).to_frame(index=False)
panel = grid.merge(panel, on=["ZIP_Code","week_start_date"], how="left")

# --- Fetch LST per ZIP (cached per ZIP) ---
lst_frames = []
for z in zips:
    sub = panel.loc[panel["ZIP_Code"]==z, ["week_start_date","ZIP_Code","ZIP_Code_Location"]].drop_duplicates()
    # Geometry for this ZIP
    wkt = sub["ZIP_Code_Location"].dropna().astype(str).iloc[0] if sub["ZIP_Code_Location"].notna().any() else None
    geom = zcta_geometry(z, wkt_point=wkt)
    lst_df = weekly_lst_for_zip(sub["week_start_date"].tolist(), geom)
    lst_df["ZIP_Code"] = z
    lst_frames.append(lst_df[["ZIP_Code","week_start_date","lst_day_c"]])

lst_panel = pd.concat(lst_frames, ignore_index=True)
data = panel.merge(lst_panel, on=["ZIP_Code","week_start_date"], how="left")
data.rename(columns={"ILI_Activity_Level":"ili"}, inplace=True)

# Drop rows missing either series
data = data.dropna(subset=["ili","lst_day_c"]).sort_values(["ZIP_Code","week_start_date"])

# --- Train/val/test split by TIME (same boundaries for all ZIPs) ---
train_mask = data["week_start_date"] <= TRAIN_END
test_mask  = data["week_start_date"] >= TEST_START

train_df = data.loc[train_mask].copy()
test_df  = data.loc[test_mask].copy()

# validation = last 20% of train for each ZIP to keep temporal ordering
def split_zip(df_zip):
    n = len(df_zip)
    v = max(int(0.2*n), WINDOW+5)
    return df_zip.iloc[:n-v], df_zip.iloc[n-v:]

trn_parts, val_parts = [], []
for z, g in train_df.groupby("ZIP_Code"):
    a, b = split_zip(g)
    trn_parts.append(a); val_parts.append(b)

trn_df = pd.concat(trn_parts); val_df = pd.concat(val_parts)

# --- Scaling: fit on TRAIN ONLY (all zips pooled) ---
from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

def pack_features(df):
    return df[["ili","lst_day_c"]].values

X_trn = scaler_X.fit_transform(pack_features(trn_df))
X_val = scaler_X.transform(pack_features(val_df))
X_tst = scaler_X.transform(pack_features(test_df))

y_trn = scaler_y.fit_transform(trn_df[["ili"]]).ravel()
y_val = scaler_y.transform(val_df[["ili"]]).ravel()
y_tst = scaler_y.transform(test_df[["ili"]]).ravel()

# --- Build sequences per ZIP to respect temporal order ---
def make_sequences_by_zip(df, X_scaled, y_scaled):
    Xs, ys = [], []
    idx = df.reset_index(drop=True).index
    # map from global row idx to position in X_scaled/y_scaled
    # We rely on the same ordering of df and X_scaled arrays
    # (we constructed X_scaled from df in that order)
    start = 0
    for z, g in df.groupby("ZIP_Code"):
        n = len(g)
        Xg = X_scaled[start:start+n, :]
        yg = y_scaled[start:start+n]
        start += n
        for i in range(n - WINDOW):
            Xs.append(Xg[i:i+WINDOW, :])
            ys.append(yg[i+WINDOW])
    return np.asarray(Xs), np.asarray(ys)

X_trn_seq, y_trn_seq = make_sequences_by_zip(trn_df, X_trn, y_trn)
X_val_seq, y_val_seq = make_sequences_by_zip(val_df, X_val, y_val)
X_tst_seq, y_tst_seq = make_sequences_by_zip(test_df, X_tst, y_tst)

print("Sequences -> Train:", X_trn_seq.shape, "Val:", X_val_seq.shape, "Test:", X_tst_seq.shape)

# --- Model per your spec ---
tf.keras.backend.clear_session()
model = models.Sequential([
    layers.Input(shape=(WINDOW, 2)),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(1, activation="linear")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_trn_seq, y_trn_seq,
          validation_data=(X_val_seq, y_val_seq),
          epochs=100, batch_size=32, callbacks=[es], verbose=1)

# --- Evaluate on TEST ---
y_pred_scaled = model.predict(X_tst_seq).reshape(-1,1)
y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
y_true = scaler_y.inverse_transform(y_tst_seq.reshape(-1,1)).ravel()
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"TEST RMSE: {rmse:.3f} | TEST MAE: {mae:.3f}")
