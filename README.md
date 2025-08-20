# Prediksi & Deteksi Anomali Data Logger AWLR Papringan

Proyek ini dikembangkan untuk melakukan **prediksi** dan **deteksi anomali** pada data hasil pencatatan **AWLR (Automatic Water Level Recorder) Papringan**.  
Sistem ini memanfaatkan model pembelajaran mesin (ML) berbasis time-series untuk menganalisis data tinggi muka air (TMA) dan sensor terkait.

---

## Tujuan
- Melakukan **prediksi TMA** (Tinggi Muka Air) beberapa langkah ke depan.
- Mengidentifikasi **anomali** pada data logger yang berpotensi disebabkan oleh:
  - Gangguan perangkat
  - Kesalahan pengukuran
  - Kejadian hidrologis ekstrem (banjir, debit tinggi)
- Membantu proses **early warning system** untuk mitigasi banjir.

---

## Fitur Utama
- **Preprocessing Data**  
  Membersihkan data logger dari missing values, outlier, dan noise.

- **Model Prediksi**  
  Menggunakan model **TCN**  untuk memprediksi data Ketinggiang Muka Air 60 menit ke depan.

- **Deteksi Anomali**  
  Menggunakan metode statistik (threshold, rolling window) maupun AI (Isolation Forest, clustering).

---

## Struktur Proyek
