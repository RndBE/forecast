import onnx
import onnxruntime as ort

# === Load dan cek struktur model ===
model_path = "tcn_mimo.onnx"

# Load model
model = onnx.load(model_path)

# Validasi apakah model ONNX valid
onnx.checker.check_model(model)
print("✅ Model ONNX valid.")

# === Cek input & output dari model ===
session = ort.InferenceSession(model_path)

print("\n=== Input Info ===")
for inp in session.get_inputs():
    print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\n=== Output Info ===")
for out in session.get_outputs():
    print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
