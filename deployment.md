# Deployment — Streamlit + Jupyter Book

> **Status:** aplikasi Streamlit **belum di-deploy**. Halaman ini adalah petunjuk; klaim "sudah online"
> hanya boleh diberikan setelah deploy berhasil dilakukan.

## Aplikasi Streamlit (`app.py`)

### Berkas yang dibutuhkan (ringan, muat untuk GitHub)

| Berkas | Ukuran |
|---|---|
| `app.py`, `requirements.txt` | kecil |
| `outputs/figures/klasifikasi_overlay.png` | ~0,3 MB |
| `outputs/figures/overlay_bounds.json` | <1 KB |
| `outputs/tables/*.csv` (metrik, CI, luas, split, validity) | <1 MB |
| `outputs/models/pixel_svm.pkl` + `metadata.json` | ~1,1 MB |
| `data/boundary_jatim.geojson` | ~1 MB |

**Jangan** mengunggah file besar berikut ke GitHub tanpa Git LFS (melebihi limit 100 MB):
`pixel_et.pkl` (±1,3 GB), `pixel_rf.pkl` / `random_forest_best.pkl` (±0,5 GB), `*.tif` raster,
`outputs/tables/*.pkl`/`*.npz`, dan `outputs/figures/folium_polygon_map.html` bila diperlukan.
App hanya membutuhkan `pixel_svm.pkl`; file lain hanya untuk reproducibility lokal.

### Langkah deploy (Streamlit Community Cloud)

1. Push repo ke GitHub (pastikan `.gitignore`/LFS menangani file > 100 MB seperti di atas).
2. Buka <https://share.streamlit.io> → **New app** → pilih repo, branch, `app.py` sebagai Main file path.
3. Streamlit memakai `requirements.txt` otomatis; klik **Deploy**.
4. Setelah sukses, bar URL-nya di sini (diisi setelah deploy — jangan mengklaim online sebelum ini).

### Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Isi aplikasi

- **Ringkasan:** 8 kartu angka kunci (akurasi/F1 test, CV pemenang, jumlah polygon, pixel valid, luas)
  + bar chart F1 per model + unduh CSV + catatan teknis (peta precomputed vs confusion dihitung ulang).
- **Peta klasifikasi:** toggle basemap Esri/Google, opacity slider, legenda 6 kelas, boundary provinsi,
  plus perbandingan RGB vs hasil klasifikasi (4 zoom uji).
- **Luas per kelas:** tabel + bar chart dari `luas_per_kelas.csv` (Tahap 5).
- **Evaluasi model:** tabel test-vs-CV, bootstrap CI 90%, confusion matrix (dihitung ulang dari
  `testing_pixel.csv` + `pixel_svm.pkl` — bukan angka hardcode) + F1 per kelas + learning curve.
- **Metodologi:** ringkasan data, profil spektral, provenance label, keterbatasan, jumlah polygon.
- **Peringatan >100 MB** tampil otomatis bila file besar terdeteksi.

## Jupyter Book (web statis)

```bash
pip install jupyter-book
jupyter-book build .
```

Hasil ada di `_build/html/`. Konfigurasi: `execute_notebooks: off` (output sudah tersimpan di `.ipynb`),
`exclude_patterns` menyingkirkan raster/model/data mentah dari build.
