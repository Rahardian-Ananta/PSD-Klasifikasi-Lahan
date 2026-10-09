# Panduan Deploy ke GitHub

Status: repo **belum** dibuat (belum ada `.git` di folder ini). Panduan ini memakai ukuran file
aktual per **9 Okt 2026**. Limit GitHub: **100 MB per file** (peringatan >50 MB), direkomendasikan
repo total < 1 GB.

## 1. Daftar file

### WAJIB (book + Streamlit app) ± 25 MB

| Path | Ukuran | Untuk apa |
|---|---|---|
| `notebooks/01…06_*.ipynb` | 6,8 MB | isi buku (sudah ter-eksekusi) |
| `app.py`, `requirements.txt`, `deployment.md` | <10 KB | aplikasi Streamlit |
| `_toc.yml`, `_config.yml`, `intro.md`, `logo.png`, `references.bib` | kecil | konfigurasi buku |
| `markdown.md`, `markdown-notebooks.md`, `panduan_*.md` | kecil | lampiran |
| `prompt/` | kecil | dokumentasi aturan project |
| `outputs/figures/` | 5,8 MB | semua PNG + overlay + `webgis.html` |
| `outputs/tables/*.csv` + `*.geojson` + `*.json` | ±10 MB | metrik, luas, split, statistik polygon |
| `outputs/tables/training_pixel.csv`, `testing_pixel.csv` | 5,8 MB | data latih/uji (dipakai app) |
| `outputs/models/pixel_svm.pkl` + `best_model.pkl` + `metadata.json` | 2,2 MB | model pemenang (dipakai app) |
| `data/boundary_jatim.geojson` | 152 KB | batas provinsi (dipakai app) |
| `data/s2_jatim_metadata.json`, `data/sumber_data_*.ipynb/py`, `training_samples_summary.csv` | kecil | provenance |

### OPSIONAL (boleh, masih muat)

| Path | Ukuran | Catatan |
|---|---|---|
| `outputs/raster/klasifikasi_jawa_timur.tif` | 44 MB | hasil peta final; di bawah limit tapi besar — boleh dilewati |
| `data/training_samples_jawa_timur.geojson/.gpkg` | 16 MB | sample sumber, tidak dipakai app/book |
| `outputs/models/mean_*.pkl`, `pixel_lgbm.pkl`, `pixel_logreg.pkl` | ±18 MB | model lain untuk reproducibility |

### JANGAN DIUNGGAH (melebihi limit / tidak perlu)

| Path | Ukuran | Alasan |
|---|---|---|
| `data/s2_jatim.tif` | **1,56 GB** | raster Sentinel-2 mentah — re-download via GEE (`sumber_data_01.py`); melebihi kuota LFS |
| `outputs/models/pixel_et.pkl` | **1,31 GB** | dibuat ulang di notebook 04; melebihi kuota LFS |
| `outputs/tables/*.pkl`, `*.npz` | ±12 MB | cache, bisa dibangun ulang dari notebook |
| `data/geoboundaries_IDN_ADM1.geojson` | 15 MB | sumber yang dilarang dipakai project |
| `_build/` | 38 MB | hasil build, bisa dibuat ulang |

### Lewat Git LFS (`.gitattributes` sudah disediakan)

`pixel_rf.pkl` dan `random_forest_best.pkl` (2× 474 MB) **tidak muat di git biasa** (> 100 MB)
tetapi tetap ingin disimpan → diarahkan ke **Git LFS**. Total 946 MB, masih di bawah kuota
gratis 1 GB. File `s2_jatim.tif` (1,5 GB) dan `pixel_et.pkl` (1,3 GB) **tidak** ikut LFS
karena akan melewati kuota.

> Catatan kuota: GitHub LFS gratis = **1 GB penyimpanan + 1 GB bandwidth/bulan**. Satu kali
> clone penuh menghabiskan bandwidth; **Streamlit Community Cloud tidak menjamin menarik objek
> LFS**, tetapi app ini tidak memakainya. Bila kuota jadi masalah, pindahkan kedua `.pkl` ke
> **GitHub Release asset** dan hapus barisnya dari `.gitattributes`.

## 2. `.gitignore` dan `.gitattributes`

Kedua file sudah ada di repo. Untuk mengaktifkan LFS sekali saja:

```text
git lfs install
```

Isi `.gitignore` (dikecualikan dari repo):

```text
data/s2_jatim.tif
data/geoboundaries_IDN_ADM1.geojson
outputs/models/pixel_et.pkl
outputs/tables/*.pkl
outputs/tables/*.npz
_build/
__pycache__/
.ipynb_checkpoints/
.venv/
```

Isi `.gitattributes` (disimpan via LFS):

```text
outputs/models/pixel_rf.pkl filter=lfs diff=lfs merge=lfs -text
outputs/models/random_forest_best.pkl filter=lfs diff=lfs merge=lfs -text
```

## 3. Inisialisasi dan push

```bash
# dari folder D:\book
git init
git add .
git commit -m "Project klasifikasi lahan Jawa Timur: 6 notebook + WebGIS + Streamlit"

# cek dulu tidak ada file besar yang ikut (harus kosong):
git ls-files -s | grep -E "\.(tif|pkl|npz)$" | while read m h i f; do
  sz=$(stat -c%s "$f" 2>/dev/null); [ "$sz" -gt 104857600 ] && echo "LEBIH 100MB: $f ($sz)"
done

# buat repo di github.com (New repository, JANGAN centang README/.gitignore)
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

Alternatif tanpa terminal: buat repo di GitHub → **Add file → Upload files** → seret folder
(raster/model yang dilarang otomatis ditolak bila >100 MB).

## 4. Verifikasi setelah push

- [ ] Buka repo di GitHub: tidak ada file merah/lebih 100 MB.
- [ ] `notebooks/05_evaluation.ipynb` dan `06_webgis.ipynb` tampil dengan output (bukan kosong).
- [ ] `app.py`, `requirements.txt`, `deployment.md` ada.
- [ ] Ukuran repo: `git count-objects -H` (target < 100 MB).

## 5. Lanjutan

1. **Streamlit Community Cloud** — langkah lengkap ada di [`deployment.md`](deployment.md).
   App hanya butuh daftar WAJIB; setelah deploy, tulis URL-nya di `deployment.md`
   (jangan klaim online sebelum berhasil).
2. **Jupyter Book online (opsional)** — GitHub Pages:
   ```bash
   pip install jupyter-book
   jupyter-book build .
   # unggah isi _build/html ke branch gh-pages, atau pakai GitHub Action jb-publish
   ```

## 6. Kalau file besar tetap perlu di-repo

Gunakan [Git LFS](https://git-lfs.com) (limit file LFS 2 GB/bulan bandwidth):
`git lfs install && git lfs track "*.tif"` lalu tambahkan `.gitattributes` sebelum `git add`.
Cara paling ringan: unggah file besar sebagai **GitHub Release asset** (tanpa limit bermakna)
dan tulis URL unduhannya di `deployment.md`.
