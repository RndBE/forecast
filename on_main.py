# app.py
import io, json, joblib, numpy as np, pandas as pd
from flask import Flask, request, jsonify
import onnxruntime as ort

# ========= Helpers =========
def make_windows(arr, w):
    # arr: (N,), return: (num_win, w)
    arr = np.asarray(arr, dtype=np.float32)
    return np.stack([arr[i:i+w] for i in range(len(arr)-w+1)], axis=0) if len(arr) >= w else np.empty((0, w), np.float32)

def align_scores(idx, scores, w):
    s = pd.Series(np.nan, index=idx)
    if len(scores) > 0:
        s.iloc[w-1:w-1+len(scores)] = scores
    return s

def load_input_from_request(req):
    """
    Prioritas:
    1) multipart CSV (field 'file')
    2) JSON dengan key 'sensor1' (+ optional 'timestamps')
    """
    if "file" in req.files:
        f = req.files["file"]
        df = pd.read_csv(f)
        # Nama kolom minimal: sensor1
        if "sensor1" not in df.columns:
            raise ValueError("CSV harus punya kolom 'sensor1'.")
        # Index waktu opsional
        if "Waktu" in df.columns:
            df["Waktu"] = pd.to_datetime(df["Waktu"], errors="coerce")
            df = df.dropna(subset=["Waktu"])
            df = df.set_index("Waktu").sort_index()
        return df[["sensor1"]].astype(float)

    data = req.get_json(force=True, silent=False)
    if not data or "sensor1" not in data:
        raise ValueError("JSON harus berisi key 'sensor1' (list angka).")

    s = pd.Series(data["sensor1"], dtype="float32")
    if "timestamps" in data and data["timestamps"]:
        idx = pd.to_datetime(pd.Series(data["timestamps"]), errors="coerce")
        df = pd.DataFrame({"sensor1": s.values}, index=idx)
        df = df[~df.index.isna()]
        if len(df) == 0:
            # fallback ke range index jika semua timestamp invalid
            df = pd.DataFrame({"sensor1": s.values})
    else:
        df = pd.DataFrame({"sensor1": s.values})
    return df

# ========= App =========
app = Flask(__name__)

# Load artefak
with open("tma_config.json") as f:
    cfg = json.load(f)
WINDOW   = cfg["window"]
THRESH   = cfg["threshold"]
HIDDEN   = cfg["hidden"]

scaler = joblib.load("tma_scaler.pkl")

# ONNX Runtime session
sess = ort.InferenceSession(
    "tma_lstm_ae.onnx",
    providers=["CPUExecutionProvider"]
)
inp_name  = sess.get_inputs()[0].name
out_name  = sess.get_outputs()[0].name

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "window": WINDOW, "threshold": THRESH}, 200

import requests

def post_anomali(url: str, waktu: str, sensor1: float, status_anomali: str, timeout: int = 10):
    """
    Kirim hasil anomali ke endpoint receiver (PHP) via HTTP POST (form-data).
    """
    payload = {
        "waktu": waktu,
        "sensor1": sensor1,
        "status_anomali": status_anomali
    }
    try:
        resp = requests.post(url, data=payload, timeout=timeout)
        return (resp.ok, resp.status_code, resp.text)
    except requests.RequestException as e:
        return (False, None, f"Request error: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        df = load_input_from_request(request)
        df = df.copy()

        # --- Urutkan waktu ascending jika ada timestamp ---
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        elif "Waktu" in df.columns:
            df["Waktu"] = pd.to_datetime(df["Waktu"], errors="coerce")
            df = df.dropna(subset=["Waktu"]).sort_values("Waktu").set_index("Waktu")

        # Drop NaN & pastikan float32
        df["sensor1"] = pd.to_numeric(df["sensor1"], errors="coerce")
        df = df.dropna(subset=["sensor1"])
        if len(df) < WINDOW:
            return jsonify({"error": f"Data terlalu pendek. Minimal {WINDOW} titik."}), 400

        # Scale
        scaled = scaler.transform(df[["sensor1"]]).astype(np.float32).squeeze()

        # Windowing
        X_all = make_windows(scaled, WINDOW).astype(np.float32)
        X_all_3d = X_all[..., None]  # (num_win, WINDOW, 1)

        # Inference ONNX
        outputs = sess.run([out_name], {inp_name: X_all_3d})[0].astype(np.float32)
        recon = outputs.squeeze(-1)  # (num_win, WINDOW)

        # Recon error (MAE per-window)
        err = np.mean(np.abs(recon - X_all), axis=1)

        # Align ke index asli
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(len(df))
        err_s = align_scores(idx, err, WINDOW)

        # Labeling
        anom_ae   = (err_s > THRESH).fillna(False)
        anom_zero = (df["sensor1"] <= 0)
        anom_final = (anom_zero | anom_ae).astype(bool)

        out = pd.DataFrame({
            "sensor1": df["sensor1"].values,
            "recon_error": err_s.values,
            "anom_zero": anom_zero.values,
            "anom_ae": anom_ae.values,
            "anom_final": anom_final.values
        }, index=idx)

        # Ambil baris terakhir
        last_df = out.tail(1).reset_index(
            names="timestamp" if isinstance(idx, pd.DatetimeIndex) else "row"
        )
        if "timestamp" in last_df.columns:
            last_df["timestamp"] = pd.to_datetime(last_df["timestamp"], errors="coerce") \
                                        .dt.strftime("%Y-%m-%d %H:%M:%S")

        last_df["anom_final"] = last_df["anom_final"].map({True: "anomaly", False: "Normal"})

        # Data yang dikembalikan ke client
        result = {
            "timestamps": last_df["timestamp"].iloc[0],
            "sensor1": float(last_df["sensor1"].iloc[0]),
            "anom_final": last_df["anom_final"].iloc[0]
        }

        # === Kirim ke PHP endpoint ===
        forward_url = "http://bbwsso.monitoring4system.com/datamasuk/tes_anomali"
        ok, code, text = post_anomali(
            url=forward_url,
            waktu=result["timestamps"],
            sensor1=result["sensor1"],
            status_anomali=result["anom_final"]
        )
        print(f"[FORWARD] ok={ok}, code={code}, resp={text}")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Gunakan host=0.0.0.0 kalau mau diakses dari jaringan
    app.run(host="0.0.0.0", port=8000, debug=False)
