# app.py — Flask API: Forecast (TCN/ONNX) + Check Anomaly (LSTM-AE/ONNX)
# --------------------------------------------------------------------
# Endpoint:
#   - POST /forecast        → Prediksi ke depan; output HANYA {'waktu','sensor1'}
#   - POST /check_anomaly   → Deteksi anomali sensor1 (baris terakhir)
#
# Input format (keduanya mendukung):
#   1) JSON: {"data": [{"waktu":"...","sensor1":...,"sensor2":...}, ...]}   # /forecast
#      JSON: {"sensor1": [...], "timestamps": ["..."]}                          # /check_anomaly
#   2) CSV upload (multipart, field 'file'). Kolom minimal:
#      - /forecast: waktu, sensor1, sensor2
#      - /check_anomaly: sensor1 (opsional waktu/Waktu)
#
# Jalankan:
#   pip install flask onnxruntime pandas numpy python-dateutil joblib
#   python app.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dateutil import parser as dateparser
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os, json, warnings
warnings.filterwarnings('ignore')

# ======== KONFIGURASI GLOBAL ========
# --- Forecast (TCN) ---
ONNX_PATH   = os.environ.get("ONNX_PATH", "tcn_mimo_2.onnx")
IN_WIN      = int(os.environ.get("IN_WIN", 60))
OUT_WIN     = int(os.environ.get("OUT_WIN", 60))
BASE_COLS   = ("sensor1", "sensor2")  # kolom input dasar
SCALER_MU   = os.environ.get("SCALER_MU", "mu.npy")   # shape (C_in,)
SCALER_SD   = os.environ.get("SCALER_SD", "sd.npy")   # shape (C_in,)
PROVIDERS   = os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider")

# --- Anomaly (LSTM-AE) ---
AE_CONFIG_PATH = os.environ.get("AE_CONFIG_PATH", "tma_config.json")
AE_SCALER_PATH = os.environ.get("AE_SCALER_PATH", "tma_scaler.pkl")
AE_ONNX_PATH   = os.environ.get("AE_ONNX_PATH", "tma_lstm_ae.onnx")

# ======== LOAD ONNXRUNTIME ========
try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit("Module 'onnxruntime' belum terpasang. Jalankan: pip install onnxruntime") from e

# --- Session untuk Forecast ---
onnx_path = Path(ONNX_PATH)
if not onnx_path.exists():
    raise FileNotFoundError(f"Model ONNX (forecast) tidak ditemukan: {onnx_path.resolve()}")

providers = [p.strip() for p in PROVIDERS.split(",") if p.strip()]
sess_fore = ort.InferenceSession(str(onnx_path), providers=providers)

# --- Artefak untuk Anomaly ---
config_path = Path(AE_CONFIG_PATH)
if not config_path.exists():
    raise FileNotFoundError(f"Config AE tidak ditemukan: {config_path.resolve()}")
with open(config_path, "r") as f:
    cfg_ae = json.load(f)

WINDOW_AE = int(cfg_ae.get("window"))
THRESH_AE = float(cfg_ae.get("threshold"))
# 'hidden' boleh ada/tidak; tidak dipakai langsung di inference

scaler_ae_path = Path(AE_SCALER_PATH)
if not scaler_ae_path.exists():
    raise FileNotFoundError(f"Scaler AE tidak ditemukan: {scaler_ae_path.resolve()}")
scaler_ae = joblib.load(str(scaler_ae_path))

ae_onnx_path = Path(AE_ONNX_PATH)
if not ae_onnx_path.exists():
    raise FileNotFoundError(f"Model AE ONNX tidak ditemukan: {ae_onnx_path.resolve()}")

sess_ae = ort.InferenceSession(str(ae_onnx_path), providers=["CPUExecutionProvider"])  # AE biasanya ringan di CPU
inp_name_ae  = sess_ae.get_inputs()[0].name
out_name_ae  = sess_ae.get_outputs()[0].name

# ======== UTIL UMUM ========
def _to_datetime(x) -> datetime:
    if isinstance(x, datetime):
        return x
    return dateparser.parse(str(x))


def add_roc(df: pd.DataFrame, base_cols: Tuple[str, str] = BASE_COLS) -> pd.DataFrame:
    df = df.copy()
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom wajib tidak ada: {c}")
        df[f"{c}_RoC"] = df[c].diff().fillna(0.0)
    return df


def interleave_feats(df: pd.DataFrame, order: Tuple[str, ...] = ("sensor1","sensor1_RoC","sensor2","sensor2_RoC")) -> Tuple[np.ndarray, List[str]]:
    feat_names = list(order)
    feats = df[feat_names].to_numpy(dtype=np.float32)
    return feats, feat_names


def load_scaler(c_in: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mu = sd = None
    if Path(SCALER_MU).exists() and Path(SCALER_SD).exists():
        mu = np.load(SCALER_MU).astype(np.float32)
        sd = np.load(SCALER_SD).astype(np.float32)
        if mu.shape != (c_in,) or sd.shape != (c_in,):
            mu = sd = None
    return mu, sd


def fit_batch_scaler(feats_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats_raw.mean(axis=0)
    sd = feats_raw.std(axis=0) + 1e-9
    return mu.astype(np.float32), sd.astype(np.float32)


def normalize(feats_raw: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((feats_raw - mu) / sd).astype(np.float32)


def inverse_scale(y_norm: np.ndarray, mu_t: np.ndarray, sd_t: np.ndarray) -> np.ndarray:
    mu_t = mu_t.reshape(1, -1, 1)
    sd_t = sd_t.reshape(1, -1, 1)
    return y_norm * sd_t + mu_t


def infer_onnx_forecast(x: np.ndarray) -> np.ndarray:
    return sess_fore.run([sess_fore.get_outputs()[0].name], {sess_fore.get_inputs()[0].name: x})[0]


def make_windows(arr: np.ndarray, w: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return np.stack([arr[i:i+w] for i in range(len(arr)-w+1)], axis=0) if len(arr) >= w else np.empty((0, w), np.float32)

# ======== PIPELINE FORECAST ========
def infer_pipeline_forecast(df: pd.DataFrame) -> List[Dict[str, object]]:
    df = df.copy()
    if "waktu" not in df.columns:
        raise ValueError("Kolom 'waktu' wajib ada")
    df["waktu"] = pd.to_datetime(df["waktu"]).astype("datetime64[ns]")
    df.sort_values("waktu", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = add_roc(df, base_cols=BASE_COLS)

    feats_raw, feat_names = interleave_feats(df)
    if len(feats_raw) < IN_WIN:
        raise ValueError(f"Butuh minimal {IN_WIN} baris untuk window input. Diberikan: {len(feats_raw)}")
    feats_win = feats_raw[-IN_WIN:]

    c_in = feats_win.shape[1]
    mu_file, sd_file = load_scaler(c_in)
    if mu_file is not None and sd_file is not None:
        mu_vec, sd_vec = mu_file, sd_file
    else:
        mu_vec, sd_vec = fit_batch_scaler(feats_win)

    feats_norm = normalize(feats_win, mu_vec, sd_vec)

    x = feats_norm.T[None, ...]  # (1, C_in, IN_WIN)

    y_norm = infer_onnx_forecast(x)  # [1, 1, OUT_WIN]

    target_idx = 0  # sensor1
    mu_t = mu_vec[target_idx:target_idx+1]
    sd_t = sd_vec[target_idx:target_idx+1]
    y_pred = inverse_scale(y_norm, mu_t, sd_t)[0, 0, :]

    times = df["waktu"].to_numpy()
    if len(times) >= 2:
        deltas = np.diff(times).astype("timedelta64[s]").astype(np.int64)
        step_s = int(np.median(deltas)) if len(deltas) else 60
        step_s = 60 if step_s <= 0 else step_s
    else:
        step_s = 60

    t0 = times[-1]
    waktu_out = [pd.Timestamp(t0) + pd.to_timedelta((i+1) * step_s, unit="s") for i in range(OUT_WIN)]

    results = []
    for t, val in zip(waktu_out, y_pred.astype(float).tolist()):
        results.append({"waktu": str(t), "sensor1": val})

    return results

# ======== PIPELINE CHECK ANOMALY ========
def infer_pipeline_anomaly(df: pd.DataFrame) -> Dict[str, object]:
    df = df.copy()
    # dukung nama waktu opsional ('waktu' atau 'Waktu')
    time_col = None
    if "waktu" in df.columns:
        time_col = "waktu"
    elif "Waktu" in df.columns:
        time_col = "Waktu"

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # wajib sensor1
    if "sensor1" not in df.columns:
        raise ValueError("Kolom 'sensor1' wajib ada untuk anomaly check")

    df["sensor1"] = pd.to_numeric(df["sensor1"], errors="coerce")
    df = df.dropna(subset=["sensor1"])  # buang NaN

    if len(df) < WINDOW_AE:
        raise ValueError(f"Data terlalu pendek untuk anomaly. Minimal {WINDOW_AE} titik, hanya {len(df)}.")

    # scale → window → AE
    scaled = scaler_ae.transform(df[["sensor1"]].values).astype(np.float32).squeeze()

    X_all = make_windows(scaled, WINDOW_AE).astype(np.float32)
    X_all_3d = X_all[..., None]  # (num_win, WINDOW, 1)

    outputs = sess_ae.run([out_name_ae], {inp_name_ae: X_all_3d})[0]
    recon = outputs.squeeze(-1) if outputs.ndim == 3 else outputs  # (num_win, WINDOW)

    if recon.shape[:2] != X_all.shape[:2]:
        raise ValueError(f"Bentuk output AE tidak sesuai. Dapat {recon.shape}, harap (num_win, WINDOW[, 1]).")

    err = np.mean(np.abs(recon - X_all), axis=1)  # MAE per window

    # ambil baris terakhir
    last_val = float(df["sensor1"].iloc[-1])
    last_ts  = df[time_col].iloc[-1] if time_col else None
    last_err = float(err[-1])
    is_anom  = (last_err > THRESH_AE) or (last_val <= 0)

    return {
        "timestamp": str(last_ts) if last_ts is not None else str(len(df)-1),
        "sensor1": last_val,
        "anom_final": "anomaly" if is_anom else "Normal"
    }

# ======== FLASK APP ========
app = Flask(__name__)

@app.route("/health", methods=["GET"])  # kesehatan sederhana
def health():
    return jsonify({
        "status": "ok",
        "model_forecast": Path(ONNX_PATH).name,
        "providers": providers,
        "ae": {
            "config": Path(AE_CONFIG_PATH).name,
            "scaler": Path(AE_SCALER_PATH).name,
            "model": Path(AE_ONNX_PATH).name,
            "window": WINDOW_AE,
            "threshold": THRESH_AE,
            "warmup": WINDOW_AE - 1
        }
    })

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        if request.content_type and request.content_type.startswith("application/json"):
            payload = request.get_json(silent=True) or {}
            data = payload.get("data")
            if not isinstance(data, list) or len(data) == 0:
                return jsonify({"error": "Body JSON harus memiliki key 'data' berupa list berisi objek {waktu,sensor1,sensor2}."}), 400
            df = pd.DataFrame(data)
        elif "file" in request.files:
            f = request.files["file"]
            df = pd.read_csv(f, parse_dates=["waktu"])  # kolom wajib: waktu,sensor1,sensor2
        else:
            return jsonify({"error": "Kirim JSON (Content-Type: application/json) dengan key 'data' atau upload CSV lewat field 'file'."}), 400

        need = {"waktu", "sensor1", "sensor2"}
        miss = need - set(df.columns)
        if miss:
            return jsonify({"error": f"Kolom wajib hilang: {sorted(list(miss))}"}), 400

        result = infer_pipeline_forecast(df)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check_anomaly", methods=["POST"])
def check_anomaly():
    try:
        # JSON (mendukung dua format):
        #   A) {"sensor1": [...], "timestamps": [...]}
        #   B) {"data": [{"waktu":..., "sensor1":..., ...}, ...]}  ← seperti contoh Anda
        if request.content_type and request.content_type.startswith("application/json"):
            payload = request.get_json(silent=True) or {}
            if "sensor1" in payload:
                # Format A
                df = pd.DataFrame({"sensor1": payload["sensor1"]})
                if "timestamps" in payload:
                    df["waktu"] = pd.to_datetime(payload["timestamps"], errors="coerce")
            elif isinstance(payload.get("data"), list) and len(payload["data"]) > 0:
                # Format B (forecast-style)
                tmp = pd.DataFrame(payload["data"])
                if "sensor1" not in tmp.columns:
                    return jsonify({"error": "Payload 'data' harus memuat kolom sensor1."}), 400
                cols = [c for c in ["waktu", "sensor1"] if c in tmp.columns]
                df = tmp[cols].copy() if cols else tmp[["sensor1"]].copy()
            else:
                return jsonify({"error": "JSON tidak valid. Gunakan {'sensor1': [...], 'timestamps': [...]} atau {'data': [{waktu,sensor1,...}, ...]}"}), 400
        # CSV (field 'file')
        elif "file" in request.files:
            f = request.files["file"]
            df = pd.read_csv(f)
        else:
            return jsonify({"error": "Kirim JSON (sensor1[,timestamps] atau data[...]) atau upload CSV (field 'file')."}), 400

        result = infer_pipeline_anomaly(df)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)


