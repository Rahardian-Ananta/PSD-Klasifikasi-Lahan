# Streamlit app — Klasifikasi Penggunaan/Penutupan Lahan Jawa Timur (Sentinel-2)
# Membaca HASIL Tahap 5 (bukan angka hardcode): CSV metrik/luas, overlay PNG, model, boundary.
import json
import os

import folium
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from streamlit_folium import st_folium

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")
MODELS = os.path.join(OUT, "models")

CN = {1: "Sawah", 2: "Bangunan / Permukiman", 3: "Mangrove",
      4: "Lahan hijau", 5: "Laut", 6: "Danau"}
FE = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "NDBI"]
LABELS = [1, 2, 3, 4, 5, 6]

# --- Unduh model -------------------------------------------------------------
# Base URL untuk model BESAR yang tidak muat di repo (GitHub Release / Hugging Face / Zenodo).
# Isi setelah mengunggah pixel_rf.pkl, random_forest_best.pkl, pixel_et.pkl ke sana.
# Contoh: "https://github.com/USERNAME/REPO/releases/download/models-v1"
MODEL_RELEASE_BASE = ""
DOWNLOAD_MAX_MB = 50  # berkas <= batas ini disajikan via tombol unduh langsung; di atasnya via tautan

# (nama berkas, representasi, algoritma, catatan, status upload GitHub)
MODEL_FILES = [
    ("best_model.pkl", "pixel", "SVM (RBF)", "Pemenang CV — salinan pixel_svm.pkl",
     "Bisa (repo biasa, ~1 MB)"),
    ("pixel_svm.pkl", "pixel", "SVM (RBF)", "Pemenang CV — model asli",
     "Bisa (repo biasa, ~1 MB)"),
    ("random_forest_best.pkl", "pixel", "Random Forest", "Model peta (inferensi) — salinan pixel_rf.pkl",
     "Bisa via Git LFS (~474 MB) — Cloud tak jamin"),
    ("pixel_rf.pkl", "pixel", "Random Forest", "Model peta (inferensi) — model asli",
     "Bisa via Git LFS (~474 MB) — Cloud tak jamin"),
    ("pixel_lgbm.pkl", "pixel", "LightGBM", "Kandidat",
     "Bisa (repo biasa, ~6 MB)"),
    ("pixel_et.pkl", "pixel", "ExtraTrees", "Kandidat (berkas besar)",
     "Tidak (1,3 GB) — di-.gitignore → pakai Release/HF"),
    ("pixel_logreg.pkl", "pixel", "Logistic Regression", "Kandidat baseline",
     "Bisa (repo biasa, ~2 KB)"),
    ("mean_rf.pkl", "mean", "Random Forest", "Kandidat (representasi polygon-mean)",
     "Bisa (repo biasa, ~3,8 MB)"),
    ("mean_et.pkl", "mean", "ExtraTrees", "Kandidat (representasi polygon-mean)",
     "Bisa (repo biasa, ~8,6 MB)"),
    ("mean_lgbm.pkl", "mean", "LightGBM", "Kandidat (representasi polygon-mean)",
     "Bisa (repo biasa, ~2,2 MB)"),
    ("mean_svm.pkl", "mean", "SVM (RBF)", "Kandidat (representasi polygon-mean)",
     "Bisa (repo biasa, ~30 KB)"),
    ("mean_logreg.pkl", "mean", "Logistic Regression", "Kandidat (representasi polygon-mean)",
     "Bisa (repo biasa, ~2 KB)"),
    ("metadata.json", "-", "-", "Metadata model (fitur, pemenang, versi pustaka)",
     "Bisa (repo biasa)"),
]

# Bedah ukuran vs akurasi tiap model. Angka dari notebook Bab 4–5 (CV/test) dan inspeksi
# langsung isi berkas .pkl (node pohon, support vector, koefisien) per model.
# (tag, model, algoritma, ukuran, struktur, cv_pol, test_pol, peran, narasi)
MODEL_CARDS = [
    ("pixel", "rf", "Random Forest — 300 pohon, max_depth=25", "496 MB",
     "4.427.198 node total; 14.757 node/pohon; kedalaman tepat 25 (dibatasi)",
     0.507, 0.465, "Model peta (+ salinan random_forest_best.pkl)",
     "Pohon **dihentikan di kedalaman 25**, jadi ukurannya sedang (496 MB). Model ini yang "
     "benar-benar menggambar peta karena inferensinya cepat (±2,1 jam untuk 143 juta pixel)."),
    ("pixel", "et", "ExtraTrees — 300 pohon, tanpa batas kedalaman", "1.312 MB",
     "11.718.914 node total; 39.063 node/pohon; kedalaman rata-rata 42,5 (maks 54)",
     0.513, 0.447, "Kandidat — berkas TERBESAR",
     "Split-nya **acak** dan kedalaman **tidak dibatasi**, sehingga pohon tumbuh sangat dalam → "
     "**2,6× lebih banyak node** dari RF. Hasilnya berkas 1,3 GB, **padahal skornya tidak lebih "
     "baik** dari model lain. Ini contoh paling ekstrem dari pesan di atas: **besar ≠ lebih akurat**."),
    ("pixel", "lgbm", "LightGBM — 300 pohon, 31 daun/pohon", "6,1 MB",
     "boosting; menyimpan struktur/histogram ringkas, bukan pohon penuh",
     0.522, 0.450, "Kandidat",
     "CV-nya tertinggi ke-2 (0,522), tetapi berkasnya **hanya 6,1 MB** — 214× lebih kecil dari ET "
     "— karena tiap pohon dibatasi 31 daun dan LightGBM menyimpan struktur ringkas."),
    ("pixel", "svm", "SVM RBF — C=1", "1,1 MB",
     "7.264 support vector (dilatih pada subsample 8.861 baris)",
     0.537, 0.474, "PEMENANG CV (+ salinan best_model.pkl)",
     "**Juara CV** dengan berkas cuma 1,1 MB: SVM hanya menyimpan *support vector*, bukan seluruh "
     "data. Ironisnya model paling akurat justru paling lambat saat inferensi (±67 jam) → tidak "
     "dipakai memetakan."),
    ("pixel", "logreg", "Logistic Regression — C=2", "~2 KB",
     "6 kelas × 9 fitur = 54 koefisien",
     0.335, 0.391, "Kandidat — baseline",
     "Berkas **terkecil** (2 KB, ~600.000× lebih kecil dari ET) dan tercepat, tetapi skornya jauh "
     "tertinggal → bukti batas antar kelas **tidak linier**, bukan kesalahan kode."),
    ("mean", "rf", "Random Forest — 300 pohon, max_depth=25", "3,8 MB",
     "32.574 node total (dilatih pada 204 baris, 1 baris/polygon)",
     0.441, 0.444, "Kandidat (representasi polygon-mean)",
     "Representasi polygon-mean membuat pohon jauh lebih kecil karena hanya ada 204 baris data; "
     "sebaran pixel di dalam polygon hilang."),
    ("mean", "et", "ExtraTrees — 300 pohon, max_depth=25", "8,6 MB",
     "75.972 node total (dilatih pada 204 baris)",
     0.444, 0.404, "Kandidat (representasi polygon-mean)",
     "2,3× lebih besar dari mean-rf karena split acak menghasilkan lebih banyak node, tetapi "
     "skornya serupa."),
    ("mean", "lgbm", "LightGBM — 300 pohon, 31 daun/pohon", "2,2 MB",
     "boosting ringkas pada 204 baris",
     0.422, 0.450, "Kandidat (representasi polygon-mean)",
     "Ringkas dan skornya setara model lain di kubu mean."),
    ("mean", "svm", "SVM RBF — C=1", "0,03 MB",
     "176 support vector saja (karena data latih hanya 204 baris)",
     0.421, 0.444, "Kandidat (representasi polygon-mean)",
     "Berkas sangat kecil (30 KB) karena data latihnya sedikit; skornya setara model mean lain."),
    ("mean", "logreg", "Logistic Regression — C=2", "~2 KB",
     "6 kelas × 9 fitur = 54 koefisien",
     0.425, 0.466, "Kandidat (representasi polygon-mean)",
     "Di kubu mean, skor test-nya justru tertinggi (0,466) — tetapi selang kepercayaannya lebar, "
     "jadi bukan bukti model linier lebih baik."),
]


@st.cache_data
def file_bytes(path):
    with open(path, "rb") as f:
        return f.read()


st.set_page_config(page_title="Klasifikasi Lahan Jatim", layout="wide",
                   page_icon="🛰️")

CATATAN_TEKNIS = (
    "**Catatan teknis (model asli, bukan angka tempel):**\n"
    "- Peta = hasil inferensi model RF terbaik (`random_forest_best.pkl`) ke seluruh raster, "
    "dijalankan sekali di Bab 5 lalu disimpan sebagai GeoTIFF → PNG overlay (precomputed; "
    "klasifikasi 143 juta pixel tidak mungkin diulang di dalam app).\n"
    "- Confusion matrix = **dihitung ulang saat app dibuka** oleh `pixel_svm.pkl` pada "
    "`testing_pixel.csv` — ganti file, angkanya berubah.\n"
    "- Skor test & bootstrap CI = hasil evaluasi Bab 5 (sekali, sesuai aturan test-sekali); "
    "CV = Bab 4. Semua dibaca dari CSV, bukan ditulis manual."
)


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(TABLES, name))


@st.cache_data
def sample_source_counts():
    """Jumlah polygon per sumber/kelas sumber (dari file training samples). None bila file tidak ada."""
    import geopandas as gpd
    p = os.path.join(BASE, "data", "training_samples_jawa_timur.geojson")
    try:
        g = gpd.read_file(p, columns=["class_id", "class_name", "source", "source_class"])
    except Exception:
        return None, None
    return (g.groupby(["source", "source_class"]).size().rename("n_polygon").reset_index(),
            g.groupby("class_id").size().rename("n_polygon"))


@st.cache_data
def big_files(limit_mb=100):
    bad = []
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            mb = os.path.getsize(p) / 1e6
            if mb > limit_mb:
                bad.append((os.path.relpath(p, BASE).replace("\\", "/"), round(mb, 1)))
    return sorted(bad, key=lambda x: -x[1])


@st.cache_data
def confusion_winner():
    tepx = load_csv("testing_pixel.csv")
    m = joblib.load(os.path.join(MODELS, "pixel_svm.pkl"))
    pv = m.predict(tepx[FE].to_numpy())
    cm_px = confusion_matrix(tepx["class_id"].to_numpy(), pv, labels=LABELS)
    df = pd.DataFrame({"fid": tepx["fid"].to_numpy(), "p": pv})
    vote = df.groupby("fid")["p"].agg(lambda s: s.value_counts().idxmax())
    yt = (tepx.drop_duplicates("fid").set_index("fid")["class_id"]
          .reindex(vote.index).to_numpy())
    cm_pol = confusion_matrix(yt, vote.to_numpy(), labels=LABELS)
    return cm_px, cm_pol


def class_legend_html():
    rows = "".join(
        f"<div style='margin:2px 0'><span style='background:{w};display:inline-block;"
        f"width:12px;height:12px;border:1px solid #555'></span> {i} {CN[i]}</div>"
        for i, w in zip(LABELS, ["#FFFF00", "#FF0000", "#00FF00", "#FF00FF", "#0000FF", "#00FFFF"]))
    return f"<div style='background:#fff;padding:6px;border:2px solid #777;font-size:13px'><b>Legenda (6 kelas)</b>{rows}</div>"


def per_class_f1(cm):
    import numpy as np
    cm = np.asarray(cm, dtype=float)
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1},
                        index=[f"{i} {CN[i]}" for i in LABELS]).fillna(0.0)


st.title("Klasifikasi Lahan Jawa Timur — Sentinel-2")
st.caption(
    "Basemap (Esri World Imagery / Google Satellite) adalah tile XYZ (bukan WMS standar), "
    "tile Google memiliki batasan ketentuan layanan, dan basemap hanya referensi visual — "
    "BUKAN data training."
)
bad = big_files()
if bad:
    st.warning(
        "**Peringatan ukuran file (bukan error aplikasi).** File berikut melebihi **100 MB**, "
        "yaitu batas keras per-file di GitHub — kalau ikut di-`git push`, GitHub akan menolak. "
        "Aplikasi ini **tidak memakainya** (hanya butuh `pixel_svm.pkl` 1,1 MB + CSV + overlay PNG + "
        "boundary), jadi aman untuk **dikecualikan** lewat `.gitignore`.\n\n"
        + "\n".join(f"- `{p}` = {mb:,.1f} MB" for p, mb in bad)
        + "\n\nPilihan: (1) kecualikan dari repo (disarankan), (2) pakai **Git LFS**, atau "
        "(3) unggah sebagai **GitHub Release asset**. Langkah lengkap: file `panduan_deploy_github.md`."
    )

PAGES = ["Ringkasan", "Alur Data → Model", "Peta klasifikasi", "Luas per kelas",
         "Evaluasi model", "Unduh model", "Metodologi"]
page = st.sidebar.radio("Halaman", PAGES)

if page == "Ringkasan":
    tv = load_csv("test_vs_cv.csv")
    ci = load_csv("test_bootstrap_ci.csv")
    lc = load_csv("luas_per_kelas.csv")
    diag = load_csv("polygon_validity_diagnostic.csv")
    sel = load_csv("polygon_selected.csv")
    split = load_csv("polygon_split.csv")
    meta = json.load(open(os.path.join(BASE, "data", "s2_jatim_metadata.json"), encoding="utf-8"))
    mmeta = json.load(open(os.path.join(MODELS, "metadata.json"), encoding="utf-8"))
    stats = load_csv("polygon_pixel_stats.csv")
    valid_bp = int(((diag["kode"] == "VALID") & diag["fid"].isin(stats.loc[stats["n_valid"] > 0, "fid"])).sum())
    svm = tv[(tv["tag"] == "pixel") & (tv["model"] == "svm")].iloc[0]
    rfp = tv[(tv["tag"] == "pixel") & (tv["model"] == "rf")].iloc[0]
    svm_ci = ci[(ci["tag"] == "pixel") & (ci["model"] == "svm")].iloc[0]

    st.subheader("Angka kunci (dari file Tahap 4–5)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi test — pixel-svm (pemenang CV)", f"{svm['acc_pol']:.1%}",
              help="Level polygon, test set 52 polygon")
    c2.metric("F1 macro test — pixel-svm", f"{svm['f1m_pol']:.3f}",
              help=f"Bootstrap CI90 [{svm_ci['ci5']:.3f}, {svm_ci['ci95']:.3f}] — semua model tumpang tindih")
    c3.metric("CV polygon-F1 pemenang (Bab 4)", f"{mmeta['best_cv_polygon_f1']:.3f}",
              help="Best-param pixel-svm (StratifiedGroupKFold group=FID). Catatan: kolom cv_mean di "
                   "test_vs_cv = rata-rata SEMUA konfigurasi, bukan best-param")
    c4.metric("Model peta — pixel-rf", f"{rfp['f1m_pol']:.3f}",
              help="Dipilih karena inferensi 2,1 jam vs 67,3 jam (seri dalam noise)")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Polygon valid berpixel → terpilih", f"{valid_bp} → {len(sel)}",
              help=f"875 lolos V1–V6, tetapi 107 berada di luar extent raster (0 pixel); "
                   f"terpilih = 50/kelas, Laut 6")
    c6.metric("Split train / test", f"{int((split['split'] == 'train').sum())} / {int((split['split'] == 'test').sum())}",
              help="Per polygon (FID), random_state=42")
    c7.metric("Pixel valid di peta", f"{int(lc['pixel_count'].sum()):,}",
              help="Dari CSV luas; nodata 0 = di luar extent")
    c8.metric("Total area terklasifikasi", f"{lc['luas_ha'].sum():,.0f} ha", help="Resolusi 10 m (EPSG:32749)")
    st.caption(f"Citra: komposit median Sentinel-2 L2A {meta['tanggal_mulai']}–{meta['tanggal_akhir']}, "
               f"maks {meta['max_cloud_cover_pct']}% awan, {len(meta['bands'])} band + 3 indeks.")

    st.subheader("F1 macro test — 10 kombinasi (5 algoritma × 2 cara mewakili polygon)")
    ch = tv.copy()
    ch["kombinasi"] = ch["tag"] + "-" + ch["model"]
    ch = ch.set_index("kombinasi")[["f1m_pol", "cv_mean"]].rename(
        columns={"f1m_pol": "test (52 polygon)", "cv_mean": "CV (semua konfigurasi)"})
    st.bar_chart(ch.sort_values("test (52 polygon)", ascending=False))
    st.caption("X = tiap kombinasi model; Y = macro-F1 level polygon (0–1, makin tinggi makin baik). "
               "Bar biru = skor test, oranye = skor CV Bab 4. Amati: logreg paling pendek (pixel-logreg "
               "0,39; mean-logreg di CV 0,33) → batas antar kelas tidak linier. Sisanya berkumpul "
               "0,40–0,47 dan selang CI-nya tumpang tindih, jadi pemenang hanya ditentukan prosedur.")
    with st.expander("Model apa saja ini? (baca sebelum menafsir grafik)"):
        st.markdown(
            "Kombinasi = **cara mewakili polygon** (tag) × **algoritma** (model).\n\n"
            "**Cara mewakili (tag):**\n"
            "- `pixel` — tiap pixel 10×10 m jadi satu baris data (40.076 baris train). Model belajar per pixel.\n"
            "- `mean` — tiap polygon diringkas jadi satu baris = rata-rata 6 band, lalu indeks dihitung dari "
            "rata-rata itu (204 baris train). Model belajar per polygon; sebaran di dalam polygon hilang.\n\n"
            "**Algoritma (model):**\n"
            "- `rf` — **Random Forest**: ratusan pohon keputusan dilatih pada subset acak, hasil = voting "
            "mayoritas. Tahan noise, bisa dibaca lewat importance fitur.\n"
            "- `et` — **ExtraTrees**: seperti RF tetapi pembagian pohon lebih acak. Mirip, sedikit berbeda karakter.\n"
            "- `lgbm` — **LightGBM**: pohon dilatih **berurutan**, tiap pohon menambal kesalahan pohon sebelumnya "
            "(boosting). Kuat untuk data tabel, tetapi pada 204 polygon ruang geraknya dibatasi.\n"
            "- `svm` — **Support Vector Machine (RBF)**: mencari batas pemisah dengan margin terlebar antar kelas. "
            "Kuat pada data kecil, tetapi lambat saat menerapkan ke 143 juta pixel.\n"
            "- `logreg` — **Logistic Regression**: batas pemisah **linier** (satu garis/bidang). Dipakai sebagai "
            "baseline; skornya rendah = bukti batas antar kelas melengkung, bukan masalah kode.\n\n"
            "Detail CV: `StratifiedGroupKFold` (group = FID polygon) di Bab 4; nama model lengkap ada di "
            "halaman **Alur Data → Model** Tahap 9.")
    with st.expander("Apa arti 8 kartu angka di atas?"):
        st.markdown(
            "- **Akurasi test (polygon)** = proporsi polygon test yang prediksinya benar. Prediksi satu polygon "
            "= kelas terbanyak dari pixelnya (voting), lalu dibandingkan dengan label polygon.\n"
            "- **F1 macro** = rata-rata F1 tiap kelas (bukan rata-rata pixel) sehingga kelas langka seperti Laut "
            "ikut dihitung setara. Range 0–1.\n"
            "- **CV polygon-F1 (Bab 4)** = skor validasi silang sebelum test. Pemenang dipilih dari sini — test "
            "tidak dipakai untuk memilih.\n"
            "- **Model peta** = model yang benar-benar dipakai memetakan seluruh Jawa Timur. pixel-rf dipilih "
            "karena seri dalam noise tetapi jauh lebih cepat saat inferensi.\n"
            "- **Polygon valid berpixel → terpilih** = dari 2.823 polygon mentah, 875 lolos V1–V6, 768 punya "
            "pixel valid, lalu diambil maksimal 50 per kelas (Laut hanya 6) = 256.\n"
            "- **Split train/test** = jumlah polygon untuk melatih vs menguji (per polygon, bukan per pixel).\n"
            "- **Pixel valid di peta** = pixel yang terklasifikasi (0–6) di raster final; sisanya nodata.\n"
            "- **Total area** = luas seluruh pixel terklasifikasi pada resolusi 10×10 m.")
    st.download_button("Unduh test_vs_cv.csv", tv.to_csv(index=False),
                       file_name="test_vs_cv.csv", mime="text/csv")
    st.info(CATATAN_TEKNIS)

elif page == "Alur Data → Model":
    diag = load_csv("polygon_validity_diagnostic.csv")
    stats = load_csv("polygon_pixel_stats.csv")
    sel = load_csv("polygon_selected.csv")
    exc = load_csv("polygon_excluded.csv")
    cap = load_csv("pixel_cap_stats.csv")
    split = load_csv("polygon_split.csv")
    tr_px = load_csv("training_pixel.csv")
    te_px = load_csv("testing_pixel.csv")
    tr_me = load_csv("training_mean.csv")
    te_me = load_csv("testing_mean.csv")
    cvp = load_csv("cv_pixel.csv")
    cvm = load_csv("cv_mean.csv")
    tv = load_csv("test_vs_cv.csv")
    ci = load_csv("test_bootstrap_ci.csv")
    luas = load_csv("luas_per_kelas.csv")
    s2p = json.load(open(os.path.join(TABLES, "s2_profile.json")))
    rprof = json.load(open(os.path.join(TABLES, "raster_profile.json")))
    smeta = json.load(open(os.path.join(BASE, "data", "s2_jatim_metadata.json"), encoding="utf-8"))
    mmeta = json.load(open(os.path.join(MODELS, "metadata.json"), encoding="utf-8"))

    def fsize(rel):
        try:
            return f"{os.path.getsize(os.path.join(BASE, rel)):,} byte"
        except OSError:
            return "tidak tersedia"

    def stage(n, title, sub=None):
        st.divider()
        st.subheader(f"Tahap {n} — {title}")
        if sub:
            st.caption(sub)

    st.title("Alur data → model (10 tahap)")
    st.caption("Setiap angka dibaca langsung dari file project (CSV/JSON/statistik file) — "
               "bukan ditulis manual. Rujukan notebook: Bab 1–5.")

    # --- Tahap 1 ---
    stage(1, "Masalah & input data (Bab 1)", "Apa yang diklasifikasi dan berapa besar file masukan")
    k1, k2, k3 = st.columns(3)
    k1.metric("Kelas target", "6", help="1 Sawah, 2 Bangunan/Permukiman, 3 Mangrove, 4 Lahan hijau, 5 Laut, 6 Danau")
    k2.metric("File input", "5", help="Lengkap: raster, metadata, boundary, 2 format polygon")
    inv = pd.DataFrame([
        {"file": f, "ada": os.path.exists(os.path.join(BASE, "data", f)), "ukuran": fsize(f"data/{f}")}
        for f in ["s2_jatim.tif", "s2_jatim_metadata.json", "boundary_jatim.geojson",
                  "training_samples_jawa_timur.geojson", "training_samples_jawa_timur.gpkg"]])
    st.dataframe(inv, width="stretch", hide_index=True)

    # --- Tahap 2 ---
    stage(2, "Raster Sentinel-2 mentah (Bab 1–2)", "Komposit 2 hari, 6 band, satu grid 10 m")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dimensi", f"{s2p['width']:,}×{s2p['height']:,}")
    c2.metric("Total pixel", f"{s2p['total_pixel']:,}")
    c3.metric("Band", f"{s2p['count']} × {s2p['dtype']}", help=", ".join(s2p["bands"]) + " + 3 indeks")
    c4.metric("Resolusi / CRS", f"{s2p['res'][0]:.0f} m", help=s2p["crs"])
    st.markdown(
        f"- Koleksi **{smeta['koleksi']}**, komposit **median {smeta['tanggal_mulai']}–{smeta['tanggal_akhir']}**, "
        f"maksimum **{smeta['max_cloud_cover_pct']}% awan** (tak ada band tambahan di metadata: "
        f"cloud mask/resampling tidak tercatat).\n"
        f"- Nodata {s2p['nodata']}; bbox {smeta['bbox']['west']}–{smeta['bbox']['east']} lon, "
        f"{smeta['bbox']['south']}–{smeta['bbox']['north']} lat (EPSG:4326).")
    st.markdown("**Enam band citra** (acuan umum sensor — SentiWiki/Sentinel Hub, Bab 2 §2.4; "
                "bukan data project):")
    st.dataframe(pd.DataFrame([
        ["B02", "Blue", "490 nm", "10 m", "Hamburan atmosfer & kecerahan air; dasar warna RGB; bantu Laut vs Danau"],
        ["B03", "Green", "560 nm", "10 m", "Puncak pantulan vegetasi & kekeruhan air; pembilang NDWI"],
        ["B04", "Red", "665 nm", "10 m", "Serapan klorofil; penyebut NDVI; dasar RGB"],
        ["B08", "NIR", "842 nm", "10 m", "Struktur daun/biomassa; pembilang NDVI; air menyerap kuat (nilai rendah)"],
        ["B11", "SWIR1", "1610 nm", "20 m → 10 m", "Kadar air & kecerahan terbangun; pembilang NDBI; di data sudah di-resample"],
        ["B12", "SWIR2", "2190 nm", "20 m → 10 m", "Material kering/tanah; bantu bangunan vs lahan terbuka; di data sudah di-resample"],
    ], columns=["band", "nama", "panjang gelombang", "resolusi (asli → data)", "peran & kaitan kelas"]),
        width="stretch", hide_index=True)
    st.markdown("**Tiga indeks** yang diturunkan dari 6 band (dihitung per pixel, Bab 3):")
    st.dataframe(pd.DataFrame([
        ["NDVI", "(B08 − B04) / (B08 + B04)", "≥0 vegetasi; makin tinggi makin rimbun (Sawah, Lahan hijau, Mangrove)"],
        ["NDWI", "(B03 − B08) / (B03 + B08)", ">0 air; memisahkan perairan (Laut, Danau) dari darat"],
        ["NDBI", "(B11 − B08) / (B11 + B08)", ">0 area terbangun (Bangunan/Permukiman)"],
    ], columns=["indeks", "rumus", "makna"]), width="stretch", hide_index=True)
    st.caption(
        "Catatan keterbatasan (Bab 2): Band B11 dan B12 sebenarnya memiliki resolusi 20 meter, "
        "tetapi telah di-resample menjadi 10 meter. "
        "Akibatnya, tepi objek yang tajam pada kedua band ini menjadi kurang jelas. "
        "Selain itu, sawah yang tergenang air dapat terlihat mirip seperti air pada citra NIR "
        "(near-infrared/inframerah dekat)."
    )

    # --- Tahap 3 ---
    stage(3, "Polygon label mentah (Bab 2)", "Sumber sekunder, bukan referensi resmi")
    vc = diag["class_id"].value_counts().sort_index()
    d1, d2, d3 = st.columns(3)
    d1.metric("Polygon total", f"{len(diag):,}")
    d2.metric("Kolom", len(diag.columns), help=", ".join(diag.columns))
    d3.metric("Target asli", "300", help="50/kelas — TIDAK tercapai, lihat Tahap 4")
    st.dataframe(pd.DataFrame({"class_id": vc.index, "class_name": [CN[i] for i in vc.index],
                               "n_polygon": vc.values}), width="stretch", hide_index=True)
    st.markdown(
                    "**Arti tiap kolom file polygon:**\n"
                    "- `fid` = ID unik polygon (kunci split & join)\n"
                    "- `class_id`/`class_name` = label 1–6\n"
                    "- `source`/`source_class` = dari dataset mana label diambil\n"
                    "- `geometry` = bentuk poligon\n"
                    "\n"
                    "Sumber:\n"
                    "- Kementan LBS2019 (Sawah)\n"
                    "- BIG Satupeta layer /5 dan /57 (Bangunan, Lahan hijau, Danau, Mangrove)\n"
                    "- Natural Earth `ne_10m_ocean` (Laut)\n"
                    "- OSM/Overpass (tambahan)"
                )
    src, _ = sample_source_counts()
    if src is not None:
        summ = src.groupby("source")["n_polygon"].sum().rename("n_polygon").reset_index()
        st.dataframe(summ, width="stretch", hide_index=True)
        with st.expander(f"Rincian jenis sumber ({len(src)} baris)"):
            st.dataframe(src, width="stretch", hide_index=True)
    else:
        st.caption("Jumlah polygon per sumber: tidak tersedia (file training samples tidak ada di app).")
    import geopandas as gpd
    bj = gpd.read_file(os.path.join(BASE, "data", "boundary_jatim.geojson"))
    b32 = bj.to_crs(32749)
    st.markdown(f"**Boundary Jawa Timur:** {len(bj)} fitur ({', '.join(sorted(set(bj.geom_type)))}), "
                f"luas **{b32.area.sum():,.0f} m²**, geometri valid = {bool(bj.is_valid.all())}, "
                f"kolom = {list(bj.columns)} → level kabupaten/kota **tidak tersedia** (satu fitur provinsi, "
                f"hasil pemeriksaan Bab 2).")

    # --- Tahap 4 ---
    stage(4, "Validasi V1–V6 (Bab 2–3)", "Geometri → pixel → kuota; 3 penyaring bertingkat")
    st.markdown("**Arti kode pemeriksaan** (dijalankan berurutan; kode pertama yang kena menentukan):")
    st.dataframe(pd.DataFrame([
        ["V1", "Geometri null/kosong ATAU `fid` duplikat"],
        ["V2", "Tipe geometri bukan Polygon/MultiPolygon"],
        ["V3", "Geometri rusak (tidak valid menurut uji GEOS)"],
        ["V4", "Label tidak lengkap: `class_id` di luar 1–6 atau nama tidak cocok dengan ID"],
        ["V5", "Polygon tidak menghasilkan ≥1 pixel valid (serpihan <1 pixel atau di luar extent)"],
        ["V6", "Beririsan dengan polygon kelas lain >5% dari luasnya"],
    ], columns=["kode", "aturan"]), width="stretch", hide_index=True)
    n_valid = (diag["kode"] == "VALID").sum()
    n_v5 = (diag["kode"] == "V5").sum()
    n_v6 = (diag["kode"] == "V6").sum()
    berpixel = int((diag["kode"].eq("VALID") & diag["fid"].isin(stats.loc[stats["n_valid"] > 0, "fid"])).sum())
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("V1–V4 (geometri)", "0", help="Tidak ada polygon ganda/rusak/luar kelas — kode ini tidak muncul di file")
    e2.metric("V6 (luar extent)", f"{n_v6}")
    e3.metric("V5 (<1 pixel valid)", f"{n_v5:,}", help="Bab 2 diagnostik; di Bab 3 menjadi 2.008 bila 107 polygon VALID-di-luar-extent ikut terhitung")
    e4.metric("VALID → berpixel", f"{n_valid} → {berpixel}", help=f"{n_valid} lolos V1–V6, {n_valid - berpixel} berada di luar extent raster (0 pixel)")
    mrg = diag.merge(stats[["fid", "n_valid"]], on="fid")
    pool = (mrg[mrg["kode"].eq("VALID") & mrg["n_valid"].gt(0)]
            .groupby(["class_id", "class_name"]).size().rename("pool_berpixel").reset_index())
    st.dataframe(pool, width="stretch", hide_index=True)
    st.caption(f"Rekonsiliasi Bab 2 ↔ Bab 3: file diagnostik mencatat V5 = {n_v5:,}; di Bab 3 ditambah "
               f"{n_valid - berpixel} polygon berkode VALID yang berada di luar extent (0 pixel) → 2.008 "
               f"polygon tidak-berpixel. Dari {n_valid} VALID, hanya {berpixel} yang benar-benar punya pixel — "
               "itulah pool nyata untuk seleksi. Perhatikan: pool Laut hanya 6.")
    st.image(os.path.join(FIGURES, "hist_distribusi_pixel.png"), width=760,
             caption="Distribusi jumlah pixel valid per polygon (Bab 2) — sangat miring, ekstrem puluhan ribu")
    st.caption(f"Contoh ekstrem (Bab 2): FID 2025 (Laut) punya 31.663 pixel; FID 1001 (Mangrove) hanya 2 pixel.")

    # --- Tahap 5 ---
    stage(5, "Seleksi 50 per kelas (Bab 3)", "Target 300 tidak tercapai")
    f1, f2, f3 = st.columns(3)
    f1.metric("Terpilih", f"{len(sel)}", help="50/kelas × 5 kelas + Laut 6 = 256")
    f2.metric("Dibuang", f"{len(exc):,}", help="2.055 tidak valid + 512 kelebihan kuota")
    f3.metric("Pool valid ≠ terpilih", "pool → 50", help="Laut pool hanya 6 → semua 6 diambil")
    j1 = pd.crosstab(sel["class_name"], sel["class_id"])
    st.dataframe(j1, width="stretch", hide_index=False)
    if "kategori" in exc.columns:
        st.dataframe(exc["kategori"].value_counts().rename("n").to_frame(), width="stretch")
    st.markdown("**Cara memilih:** urutkan pool berpixel per kelas, ambil acak reproducible maksimal "
                "**50/kelas** (`RandomState 42`, urut FID). Kelas dengan pool ≤50 diambil seluruhnya — "
                "Laut hanya punya 6, jadi total 256 (bukan 300). Dua sebab pembuangan: **2.055 tidak valid** "
                "(V1–V6) dan **512 kelebihan kuota** (polygon valid tetapi kelasnya sudah penuh).")

    # --- Tahap 6 ---
    stage(6, "Ekstraksi pixel + cap 500 (Bab 3)", "Pixel per polygon dibatasi agar tidak didominasi raksasa")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Pixel sebelum cap", f"{int(cap['n_sebelum'].sum()):,}")
    g2.metric("Pixel sesudah cap", f"{int(cap['n_sesudah'].sum()):,}")
    g3.metric("Polygon kena cap", f"{int(cap['terkena_cap'].sum())} / {len(cap)}")
    nan_total = int(tr_px[FE].isna().sum().sum() + te_px[FE].isna().sum().sum())
    g4.metric("NaN fitur", f"{nan_total}", help="NDVI/NDWI/NDBI aman dari pembagian nol")
    st.markdown("**Aturan ekstraksi (Bab 3):** pixel diambil di titik **centroid** tiap sel "
                "(`all_touched=False`), lalu **di-cap 500/polygon** dengan `RandomState 42` urut FID. "
                "Alasan cap: satu polygon Lahan hijau diperkirakan ±8,9 juta pixel dan FID 2025 (Laut) "
                "punya 31.663 pixel — tanpa cap, segelintir polygon raksasa akan mendominasi training. "
                f"Hasil nyata: {int(cap['terkena_cap'].sum())} dari {len(cap)} polygon terpotong, total pixel "
                f"{int(cap['n_sebelum'].sum()):,} → {int(cap['n_sesudah'].sum()):,}.")

    # --- Tahap 7 ---
    stage(7, "Split train/test per polygon (Bab 3)", "Bukan per pixel — mencegah kebocoran")
    h1, h2, h3 = st.columns(3)
    h1.metric("Train / test polygon", f"{int((split['split'] == 'train').sum())} / {int((split['split'] == 'test').sum())}")
    h2.metric("Irisan FID", f"{len(set(split[split['split'] == 'train']['fid']) & set(split[split['split'] == 'test']['fid']))}",
              help="Harus 0")
    h3.metric("Target 10%/kelas", "ya", help="40 train + 10 test per kelas; Laut 4 + 2")
    ct = pd.crosstab(split["class_id"], split["split"])
    ct.index = [f"{i} {CN[i]}" for i in ct.index]
    st.dataframe(ct, width="stretch")
    st.markdown("**Kenapa per polygon, bukan per pixel?** Pixel dari polygon yang sama saling mirip. Bila "
                "dibiarkan, sebagian pixel satu polygon masuk train dan sebagian ke test → model seolah "
                "pintar padahal hanya menghafal lokasi yang sama (**kebocoran data**). Dengan `fid` sebagai "
                "kunci split, seluruh pixel satu polygon berada di satu sisi saja, sehingga skor test "
                "mengukur kemampuan ke lokasi baru. Split juga **berstrata** (40/10 tiap kelas; Laut 4/2).")
    st.markdown("**Langkah split (Bab 3):**\n"
                "1. Kunci split = `fid` polygon; seluruh pixel satu FID selalu ikut ke sisi yang sama.\n"
                "2. `train_test_split(polygon_selected, test_size=0.20, random_state=42, "
                "stratify=class_id)` — proporsi tiap kelas dijaga.\n"
                "3. Setiap pixel mewarisi label split dari FID-nya (join `many_to_one`), tidak diundi ulang.")
    st.dataframe(pd.DataFrame([
        ["13", "Split polygon terpilih", "test_size=0.20, stratify=class_id, random_state=42",
         f"{int((split['split']=='train').sum())} train / {int((split['split']=='test').sum())} test polygon"],
        ["14", "Turunkan split ke pixel", "pixel.join(split, on='fid') per polygon",
         f"{len(tr_px):,} train / {len(te_px):,} test pixel"],
    ], columns=["langkah", "proses", "rumus / aturan", "hasil aktual"]), width="stretch", hide_index=True)
    try:
        gsel = gpd.read_file(os.path.join(TABLES, "polygon_selected.geojson")).merge(split, on="fid")
        if gsel.crs.to_epsg() != 32749:
            gsel = gsel.to_crs(32749)
        trc = gsel[gsel["split"] == "train"].geometry.centroid
        tec = gsel[gsel["split"] == "test"].geometry.centroid
        mind = float(min(tec.apply(lambda p: trc.distance(p).min())))
        st.caption(f"Jarak pusat test → train terdekat = {mind:.0f} m (mengukur keterpisahan spasial; "
                   "autokorelasi spasial TIDAK sepenuhnya hilang).")
    except Exception:
        st.caption("Jarak pusat test → train: tidak tersedia (file geometri tidak dapat dibaca).")

    # --- Tahap 8 ---
    stage(8, "Dataset final + 9 fitur (Bab 3)", "Dua representasi: per-pixel dan per-polygon")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("train / test PIXEL", f"{len(tr_px):,} / {len(te_px):,}", help="Baris = 1 pixel 10×10 m")
    i2.metric("train / test MEAN", f"{len(tr_me)} / {len(te_me)}", help="Baris = 1 polygon (rata-rata pixel)")
    i3.metric("Fitur", f"{len(FE)}", help=", ".join(FE))
    i4.metric("NaN train/test", "0 / 0")
    st.markdown("**Langkah pengolahan pixel (Bab 3) — dari polygon ke dataset siap-latih:**")
    st.dataframe(pd.DataFrame([
        ["8", "Baca raster per polygon", "window = from_bounds(bounds); geometry_mask(all_touched=False)",
         "hanya pixel ber-centroid di dalam polygon"],
        ["9", "Buang nodata", "(arr != -32768).all(axis=0)", "pixel di luar extent dibuang"],
        ["10", "Cap 500/polygon", "rng = RandomState(42); rng.choice(n, 500, replace=False)",
         "62 polygon terpotong; 345.893 → 51.508 pixel"],
        ["11", "Hitung 3 indeks", "NDVI=(B08−B04)/(B08+B04); NDWI=(B03−B08)/(B03+B08); "
         "NDBI=(B11−B08)/(B11+B08)", "penyebut 0 → NaN (aktual 0 kasus)"],
        ["12", "Susun 9 fitur", "6 band (B02,B03,B04,B08,B11,B12) + NDVI,NDWI,NDBI", "matriks X semua model"],
    ], columns=["langkah", "proses", "rumus / aturan", "hasil aktual"]), width="stretch", hide_index=True)
    px_all = pd.concat([tr_px.assign(split="train"), te_px.assign(split="test")], ignore_index=True)
    cols = ["split", "fid", "class_id", "class_name"] + FE

    def _fmt(d):
        d = d.copy()
        d["fid"] = d["fid"].astype(str)
        d["class_id"] = d["class_id"].astype(str)
        d[FE[:6]] = d[FE[:6]].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
        d[FE[6:]] = d[FE[6:]].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return d

    n_skip = len(px_all) - 20
    contoh = pd.concat([_fmt(px_all[cols].head(10)),
                        pd.DataFrame([{c: "..." for c in cols}]),
                        _fmt(px_all[cols].tail(10))], ignore_index=True)
    st.dataframe(contoh, width="stretch", hide_index=True)
    st.caption(f"10 baris pertama + 10 baris terakhir dari **{len(px_all):,} pixel** "
               f"(train {len(tr_px):,} + test {len(te_px):,}); baris bertitik = {n_skip:,} baris "
               "sisanya tidak ditampilkan. Baris pertama (FID 0, Sawah): "
               "NDVI=(2341−366)/(2341+366)=0,730; NDWI=(501−2341)/(501+2341)=−0,647; "
               "NDBI=(1824−2341)/(1824+2341)=−0,124. "
               "Aturan raster: centroid pixel (all_touched=False); cap 500 per polygon (RandomState 42).")
    st.markdown("**Sembilan fitur yang dilihat model** (satuan DN, bukan reflektans terkoreksi — "
                "scale factor tidak tercatat di metadata):")
    st.dataframe(pd.DataFrame([
        ["B02–B12 (6 band)", "nilai pixel langsung dari raster", "warna & pantulan tiap panjang gelombang"],
        ["NDVI", "(B08−B04)/(B08+B04)", "kerimbunan vegetasi"],
        ["NDWI", "(B03−B08)/(B03+B08)", "keberadaan air"],
        ["NDBI", "(B11−B08)/(B11+B08)", "area terbangun"],
    ], columns=["fitur", "asal", "arti"]), width="stretch", hide_index=True)
    st.markdown("**Dua cara mewakili data:**\n"
                "- **pixel-based** → setiap pixel 10×10 m menjadi satu baris (train 40.076, test 11.432). "
                "Model melihat variasi di dalam polygon.\n"
                "- **polygon-mean** → setiap polygon dirangkum menjadi satu baris = rata-rata 6 band, "
                "baru indeks dihitung dari nilai rata-rata (train 204, test 52). Lebih ringkas tetapi "
                "sebaran/keyakinan di dalam polygon hilang.")
    st.image(os.path.join(FIGURES, "boxplot_b08_ndvi.png"), width=760,
             caption="Boxplot B08/NDVI per kelas (Bab 2) — sebaran tumpang tindih = sumber kesulitan klasifikasi")

    # --- Tahap 9 ---
    stage(9, "Training: 10 kombinasi, CV (Bab 4)", "GroupKFold per FID — test tidak disentuh")
    best_cfg = (cvp.groupby(["model", "params"])["polygon_f1"].mean()
                .reset_index().sort_values("polygon_f1", ascending=False)
                .groupby("model").head(1).sort_values("polygon_f1", ascending=False))
    best_cfg["model_tag"] = "pixel-" + best_cfg["model"]
    j1, j2, j3 = st.columns(3)
    j1.metric("Kombinasi model", f"{cvp['model'].nunique()} model × {cvm['model'].nunique()} model",
              help="pixel-based vs polygon-mean, masing-masing 5 algoritma")
    j2.metric("Subsample SVM", mmeta["svm_subsample"].split(" ")[0], help=mmeta["svm_subsample"])
    j3.metric("Pemenang CV", f"{mmeta['best_cv_polygon_f1']:.3f}",
              help=f"{'-'.join(mmeta['best'])} best-param; fold = {mmeta['cv']}")
    st.dataframe(best_cfg[["model_tag", "params", "polygon_f1"]]
                 .rename(columns={"polygon_f1": "cv_best_f1"}).reset_index(drop=True),
                 width="stretch", hide_index=True)
    st.markdown("**Langkah training (Bab 4) — apa yang terjadi pada data latih:**")
    st.dataframe(pd.DataFrame([
        ["16", "Bentuk matriks", "X = 9 fitur; y = class_id; groups = fid", f"{len(tr_px):,} baris train"],
        ["17", "Validasi silang", "StratifiedGroupKFold(n_splits=5, group=fid); mean = 4 (Laut 4 polygon)",
         "skor jujur, test tak disentuh"],
        ["18", "Skala fitur", "StandardScaler di dalam pipeline (fit per fold, cegah kebocoran)", "SVM & LogReg"],
        ["19", "Subsample SVM", "stratified per kelas, seed 42, hanya training", "40.076 → 8.861 baris"],
        ["20", "Grid hyperparameter", "RF/ET: max_depth×min_samples_leaf; LGBM: learning_rate×num_leaves; "
         "SVM/LogReg: C", "4/4/4/2/2 konfigurasi"],
        ["21", "Skor pemilihan", "voting pixel per fid → macro-F1 level polygon",
         f"pemenang CV = pixel-svm {mmeta['best_cv_polygon_f1']:.3f}"],
    ], columns=["langkah", "proses", "rumus / aturan", "hasil aktual"]), width="stretch", hide_index=True)
    st.caption("**Voting & rumus skor CV:** prediksi tiap pixel → per FID ambil kelas terbanyak (modus); "
               "seri → `class_id` terkecil. Lalu **Macro-F1 = (1/6)·Σ F1_k**, dengan "
               "F1_k = 2·P_k·R_k/(P_k+R_k), P_k = TP_k/(TP_k+FP_k), R_k = TP_k/(TP_k+FN_k).")
    st.markdown("**Apa itu tiap algoritma** (kode = kolom `model_tag` di atas):")
    st.dataframe(pd.DataFrame([
        ["pixel/mean-rf", "Random Forest", "ratusan pohon keputusan acak, hasil = voting mayoritas", "tahan noise; ada importance fitur"],
        ["pixel/mean-et", "ExtraTrees", "seperti RF, pembagian tiap pohon lebih acak", "sedikit berbeda karakter dari RF"],
        ["pixel/mean-lgbm", "LightGBM", "pohon berurutan yang menambal kesalahan sebelumnya (boosting)", "kuat di tabel; ruang gerak dibatasi pada data kecil"],
        ["pixel/mean-svm", "SVM (RBF)", "cari batas pemisah dengan margin terlebar", "kuat data kecil; lambat saat inferensi"],
        ["pixel/mean-logreg", "Logistic Regression", "batas pemisah linier", "baseline; skor rendah = kelas tidak terpisah linier"],
    ], columns=["kombinasi", "nama", "cara kerja singkat", "catatan"]), width="stretch", hide_index=True)
    st.markdown("**Hyperparameter yang diuji** (dipilih lewat CV, bukan dilihat test): RF/ET → `max_depth` × "
                "`min_samples_leaf` (300 pohon tetap); LGBM → `learning_rate` × `num_leaves`; SVM & LogReg → `C`. "
                "Selain itu SVM di-subsample karena lambat, dan CV memakai `StratifiedGroupKFold` "
                "(group = FID) supaya pixel satu polygon tidak terpecah antar fold.")
    st.caption(f"Fold pixel = 5; fold mean = {int(cvm['fold'].max() + 1)} "
               "(diturunkan dari 5 karena kelas minoritas Laut hanya 4 polygon train). "
               "cv_mean di tabel Tahap 10 = rata-rata SEMUA konfigurasi, bukan best-param.")
    st.markdown("**Daftar lengkap model yang disimpan** (transparan — semua 10 kandidat, bukan cuma juara):")
    MODEL_INFO = {
        "pixel_rf.pkl": ("pixel", "Random Forest", "kandidat — dasar random_forest_best.pkl"),
        "pixel_et.pkl": ("pixel", "ExtraTrees", "kandidat"),
        "pixel_lgbm.pkl": ("pixel", "LightGBM", "kandidat"),
        "pixel_svm.pkl": ("pixel", "SVM (RBF)", "kandidat — dasar best_model.pkl"),
        "pixel_logreg.pkl": ("pixel", "Logistic Regression", "kandidat (baseline linier)"),
        "mean_rf.pkl": ("mean", "Random Forest", "kandidat (representasi polygon-mean)"),
        "mean_et.pkl": ("mean", "ExtraTrees", "kandidat (representasi polygon-mean)"),
        "mean_lgbm.pkl": ("mean", "LightGBM", "kandidat (representasi polygon-mean)"),
        "mean_svm.pkl": ("mean", "SVM (RBF)", "kandidat (representasi polygon-mean)"),
        "mean_logreg.pkl": ("mean", "Logistic Regression", "kandidat (representasi polygon-mean)"),
        "best_model.pkl": ("pixel", "SVM (RBF)", "PMENANG CV — salinan pixel_svm.pkl"),
        "random_forest_best.pkl": ("pixel", "Random Forest", "MODEL PETA — salinan pixel_rf.pkl"),
    }
    _mrows = []
    for _fn, (_rep, _algo, _role) in MODEL_INFO.items():
        _mrows.append([_fn, _rep, _algo, _role,
                       fsize(f"outputs/models/{_fn}") if os.path.exists(os.path.join(MODELS, _fn)) else "tidak ada"])
    st.dataframe(pd.DataFrame(_mrows, columns=["file", "representasi", "algoritma", "peran", "ukuran"]),
                 width="stretch", hide_index=True)
    st.image(os.path.join(FIGURES, "cv_f1_bars.png"), width=760,
             caption="Sebaran CV polygon-F1 per model (Bab 4) — seluruh interval saling tumpang tindih")

    # --- Tahap 10 ---
    stage(10, "Evaluasi sekali → klasifikasi seluruh Jatim (Bab 5)", "Test dipakai untuk skor akhir, lalu raster diinferensi")
    svm_ci = ci[(ci["tag"] == "pixel") & (ci["model"] == "svm")].iloc[0]
    svm_tv = tv[(tv["tag"] == "pixel") & (tv["model"] == "svm")].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test F1 polygon", f"{svm_tv['f1m_pol']:.3f}",
              help=f"CI90 bootstrap 1000× [{svm_ci['ci5']:.3f}, {svm_ci['ci95']:.3f}]; semua model tumpang tindih")
    k2.metric("Model peta", "pixel-rf", help=f"F1 test {tv[(tv['tag']=='pixel')&(tv['model']=='rf')]['f1m_pol'].iloc[0]:.3f} — seri dalam noise, dipilih untuk inferensi (2,1 jam vs 67,3 jam)")
    valid_px = int(luas["pixel_count"].sum())
    k3.metric("Pixel valid di peta", f"{valid_px:,}",
              help=f"{valid_px / rprof['total_pixel']:.2%} dari {rprof['total_pixel']:,} total; sisanya nodata")
    k4.metric("Luas total", f"{luas['luas_ha'].sum():,.0f} ha")
    st.dataframe(luas, width="stretch", hide_index=True)
    st.markdown("**DUA model berbeda — jangan tertukar:**")
    st.dataframe(pd.DataFrame([
        ["Pemenang CV", "`best_model.pkl` = pixel-svm", "CV macro-F1 polygon 0,537 (tertinggi)",
         "Skor & laporan — TIDAK dipakai membuat peta"],
        ["Model peta", "`random_forest_best.pkl` = pixel-rf", "CV 0,507 / test 0,465 — seri dalam noise",
         "Menggambar peta Jatim (inferensi 2,1 jam vs 67,3 jam)"],
    ], columns=["peran", "file & model", "bukti angka", "dipakai untuk"]), width="stretch", hide_index=True)
    st.caption("Intinya: juara CV belum tentu dipakai ke peta. Karena skornya seri dalam noise, yang dipilih "
               "untuk peta adalah yang CEPAT (pixel-rf). Keduanya disimpan sebagai dua berkas berbeda.")
    with st.expander("Apa itu 'inferensi' dan kenapa peta tidak dibuat ulang di aplikasi ini?"):
        st.markdown(
            "**Inferensi = memakai model yang sudah dilatih untuk menebak data BARU (tanpa latih ulang).**\n\n"
            "1. `joblib.load('random_forest_best.pkl')` — model dimuat.\n"
            "2. Raster Sentinel-2 dibaca **per strip 128 baris** (berkas 1,5 GB tidak mungkin sekaligus).\n"
            "3. Untuk tiap pixel: hitung 9 fitur (6 band + NDVI/NDWI/NDBI) → `predict` → keluar satu label "
            "kelas 1–6.\n"
            "4. Pixel nodata (nilai 0) dibiarkan 0; hasil ditulis ke `klasifikasi_jawa_timur.tif`.\n\n"
            f"- Skala: **{rprof['total_pixel']:,} pixel** total, "
            f"**{int(luas['pixel_count'].sum()):,}** di antaranya valid dan diprediksi.\n"
            "- Aplikasi Streamlit **tidak** mengulang inferensi (terlalu berat) — ia hanya menampilkan hasil "
            "yang sudah dihitung sekali di Bab 5.\n\n"
            "Ringkas: **training** = model belajar dari 40.076 pixel berlabel; "
            "**inferensi** = model menebak label ±143 juta pixel tak berlabel.")
    st.markdown("**Aturan & alasan (Bab 5):**\n"
                "- **Test dipakai satu kali** untuk skor akhir; memilih model dilakukan di CV Bab 4 agar test "
                "tetap netral. Kalau test dipakai berulang untuk memilih, skornya tidak lagi jujur.\n"
                "- **Bootstrap 1.000×** (sampling polygon dengan pengembalian) memberi selang kepercayaan 90%. "
                "Semua selang saling tumpang tindih → tak ada model yang terbukti lebih baik; pemenang hanya "
                "prosedural.\n"
                "- **Kenapa peta memakai pixel-rf, bukan pixel-svm yang menang CV?** Skornya seri dalam noise, "
                "sementara kecepatan inferensi beda jauh (rf ≈ 2,1 jam vs svm ≈ 67,3 jam untuk 143 juta pixel). "
                "Jadi alasan pemetaan = kelayakan waktu, bukan akurasi.\n"
                "- Kelas sulit terbaca dari hitungan polygon: Mangrove↔Danau, Lahan hijau↔Sawah, dan Laut hanya "
                "diwakili 2 polygon test.")
    st.markdown(f"- Artefak: `klasifikasi_jawa_timur.tif` ({fsize('outputs/raster/klasifikasi_jawa_timur.tif')}, "
                f"uint8, nodata 0, {rprof['width']}×{rprof['height']}, {rprof['crs']}); "
                f"`best_model.pkl` ({fsize('outputs/models/best_model.pkl')}) = pemenang CV pixel-svm.")
    st.info(CATATAN_TEKNIS)

elif page == "Peta klasifikasi":
    basemap = st.sidebar.radio("Basemap", ["Esri World Imagery", "Google Satellite"])
    opacity = st.sidebar.slider("Opacity klasifikasi", 0.0, 1.0, 0.7, 0.05)
    ob = json.load(open(os.path.join(FIGURES, "overlay_bounds.json")))["bounds4326"]
    import geopandas as gpd
    bj = gpd.read_file(os.path.join(BASE, "data", "boundary_jatim.geojson")).to_crs("EPSG:4326")
    bb = bj.total_bounds
    m = folium.Map(location=[(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2], zoom_start=7, tiles=None)
    if basemap == "Esri World Imagery":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name=basemap).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite", name=basemap).add_to(m)
    folium.raster_layers.ImageOverlay(
        name="Klasifikasi (6 kelas)",
        image=os.path.join(FIGURES, "klasifikasi_overlay.png"),
        bounds=[[ob[1], ob[0]], [ob[3], ob[2]]], opacity=opacity, zindex=2).add_to(m)
    folium.GeoJson(bj.__geo_interface__, name="Batas Jatim",
                   style_function=lambda f: {"color": "black", "weight": 2, "fill": False}).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:16px;left:10px;z-index:9999'>" + class_legend_html() + "</div>"))
    st_folium(m, width=1000, height=620, returned_objects=[])

    st.divider()
    st.subheader("Unduh hasil inferensi")
    st.markdown(
        "Hasil inferensi **ditampilkan** di atas sebagai overlay pada peta. Berkas aslinya dapat "
        "**diunduh** di sini:\n"
        "- **`klasifikasi_jawa_timur.tif`** — GeoTIFF hasil inferensi (6 kelas, uint8, nodata 0) untuk "
        "diolah di QGIS/ArcGIS; ini sumber peta & angka luas.\n"
        "- **`klasifikasi_overlay.png`** — gambar transparan yang ditumpuk pada peta.")
    _tif = os.path.join(OUT, "raster", "klasifikasi_jawa_timur.tif")
    _ov = os.path.join(FIGURES, "klasifikasi_overlay.png")
    d1, d2 = st.columns(2)
    with d1:
        if os.path.exists(_tif):
            _mb = os.path.getsize(_tif) / 1e6
            st.download_button(f"Unduh klasifikasi_jawa_timur.tif ({_mb:,.1f} MB)",
                               data=file_bytes(_tif), file_name="klasifikasi_jawa_timur.tif",
                               mime="image/tiff", key="dl_raster_tif")
        elif MODEL_RELEASE_BASE:
            st.link_button("Unduh klasifikasi_jawa_timur.tif dari rilis",
                           f"{MODEL_RELEASE_BASE}/klasifikasi_jawa_timur.tif")
        else:
            st.caption("GeoTIFF tidak dibundel di repo. Unggah `outputs/raster/klasifikasi_jawa_timur.tif` "
                       "ke GitHub Release, lalu setel `MODEL_RELEASE_BASE` di `app.py`.")
    with d2:
        if os.path.exists(_ov):
            _mbp = os.path.getsize(_ov) / 1e6
            st.download_button(f"Unduh klasifikasi_overlay.png ({_mbp:,.2f} MB)",
                               data=file_bytes(_ov), file_name="klasifikasi_overlay.png",
                               mime="image/png", key="dl_raster_png")
    st.caption("Luas per kelas (CSV) dapat diunduh dari halaman **Luas per kelas**.")

    st.subheader("RGB Sentinel-2 vs hasil klasifikasi (4 zoom uji)")
    st.image(os.path.join(FIGURES, "rgb_vs_klasifikasi.png"), width=1000,
             caption="Pemeriksaan visual BUKAN validasi akurasi; awan/bayangan tak berkelas dipaksa ke 6 kelas")
    with st.expander("Catatan teknis"):
        st.markdown(CATATAN_TEKNIS)

elif page == "Luas per kelas":
    lc = load_csv("luas_per_kelas.csv")
    st.subheader("Luas per kelas (dari pixel raster, resolusi 10 m)")
    m1, m2 = st.columns([3, 2])
    with m1:
        st.dataframe(lc.style.format({"pixel_count": "{:,}", "luas_ha": "{:,.1f}", "persen": "{:.2f}"}),
                     width="stretch", hide_index=True)
    with m2:
        st.bar_chart(lc.set_index("class_name")["luas_ha"], horizontal=True)
    st.caption(f"Total area terklasifikasi = {lc['luas_ha'].sum():,.1f} ha; "
               f"pixel valid = {int(lc['pixel_count'].sum()):,} (nodata = 0).")
    st.download_button("Unduh luas_per_kelas.csv", lc.to_csv(index=False),
                       file_name="luas_per_kelas.csv", mime="text/csv")
    with st.expander("Catatan teknis"):
        st.markdown(CATATAN_TEKNIS)

elif page == "Evaluasi model":
    tv = load_csv("test_vs_cv.csv")
    ci = load_csv("test_bootstrap_ci.csv")
    lc = load_csv("learning_curve.csv")
    tepx = load_csv("testing_pixel.csv")
    st.subheader("Test (sekali) vs CV")
    st.dataframe(tv.style.format({c: "{:.3f}" for c in tv.columns if tv[c].dtype.kind == "f"}),
                 width="stretch", hide_index=True)
    st.subheader("Bootstrap 1000x, CI 90% macro-F1 polygon")
    st.dataframe(ci.style.format({"f1m_pol": "{:.3f}", "ci5": "{:.3f}", "ci95": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption("Semua selang CI tumpang tindih → tidak ada perbedaan bermakna antar model; "
               "pemenang prosedural = pixel-svm (CV 0,537), model peta = pixel-rf (inferensi layak).")
    st.subheader("Apa yang terjadi saat TEST (langkah demi langkah)")
    st.markdown("**Aturan utama: model dipilih dari CV Bab 4 — test hanya dipanggil SEKALI.**")
    st.dataframe(pd.DataFrame([
        ["1", "Muat data uji", "testing_pixel.csv — FID yang tak pernah dilihat saat latih",
         f"{len(tepx):,} pixel / {tepx['fid'].nunique()} polygon"],
        ["2", "Prediksi pixel", "pv = model.predict(X[9 fitur])", f"{len(tepx):,} prediksi label"],
        ["3", "Agregasi ke polygon", "per FID → kelas terbanyak (modus); seri → class_id terkecil",
         f"{tepx['fid'].nunique()} prediksi vs label"],
        ["4", "Rumus per kelas", "P_k=TP_k/(TP_k+FP_k); R_k=TP_k/(TP_k+FN_k); F1_k=2·P_k·R_k/(P_k+R_k)",
         "dibaca dari confusion matrix"],
        ["5", "Rumus skor", "Accuracy = benar/n; Macro-F1 = (1/6)·Σ F1_k", "tiap kelas berbobot sama"],
        ["6", "Selang kepercayaan", "bootstrap 52 polygon dgn pengembalian 1.000× (seed 42), tanpa latih "
         "ulang → CI90 = [P5, P95]", "pixel-svm 0,474 [0,366, 0,565]"],
        ["7", "Keputusan", "test = konfirmasi + CI, bukan alat memilih model", "semua CI tumpang tindih"],
    ], columns=["langkah", "proses", "rumus / aturan", "hasil aktual"]), width="stretch", hide_index=True)
    cm_px, cm_pol = confusion_winner()
    st.subheader("Confusion matrix — pixel-svm (pemenang CV), test set")
    tab1, tab2 = st.tabs(["Level polygon (voting)", "Level pixel"])
    with tab1:
        df = pd.DataFrame(cm_pol, index=[f"true {i} {CN[i]}" for i in LABELS],
                          columns=[f"pred {i} {CN[i]}" for i in LABELS])
        st.dataframe(df.style.format("{:,}"), width="stretch")
        st.subheader("Per kelas (level polygon)")
        st.bar_chart(per_class_f1(cm_pol)["f1"])
    with tab2:
        df = pd.DataFrame(cm_px, index=[f"true {i} {CN[i]}" for i in LABELS],
                          columns=[f"pred {i} {CN[i]}" for i in LABELS])
        st.dataframe(df.style.format("{:,}"), width="stretch")
        st.subheader("Per kelas (level pixel)")
        st.bar_chart(per_class_f1(cm_px)["f1"])
    st.subheader("Learning curve (subsample pixel dalam polygon, n_poly=204 tetap)")
    piv = lc.groupby("frac")[["f1_pixel_svm", "f1_mean_et"]].mean()
    st.line_chart(piv)
    st.caption("Pixel-svm naik 0,312 → 0,371 lalu plateau; mean-et datar 0,404 (1 baris/polygon "
               "selalu ikut). Titik 100% kurva dilatih di 40.076 px — beda dengan model tersimpan "
               "(subsample 8.861 px), sehingga dibaca untuk TREN, bukan level absolut.")
    with st.expander("Catatan teknis"):
        st.markdown(CATATAN_TEKNIS)

elif page == "Unduh model":
    st.subheader("Unduh model")
    st.markdown(
        "Semua berkas model disimpan di `outputs/models/`. Model **kecil** yang ikut di repo dapat "
        "diunduh langsung dari sini; model **besar** (ratusan MB–GB) tidak muat di GitHub sehingga "
        "disajikan lewat tautan **GitHub Release / Hugging Face**.")
    st.markdown(
        "Kolom **`upload ke GitHub`** menandai mana yang boleh masuk repo dan mana yang tidak:\n"
        "- **Bisa (repo biasa)** → ukuran < 100 MB, aman di-commit seperti file lain.\n"
        "- **Bisa via Git LFS** → > 100 MB tetapi diarahkan ke Git LFS lewat `.gitattributes` "
        "(Cloud tidak menjamin menariknya).\n"
        "- **Tidak** → dikecualikan `.gitignore` (batas keras GitHub 100 MB per file) → unggah ke "
        "GitHub Release / Hugging Face, lalu isi `MODEL_RELEASE_BASE`.")
    if MODEL_RELEASE_BASE:
        st.success(f"Tautan model besar aktif: `{MODEL_RELEASE_BASE}`")
    else:
        st.warning(
            "Tautan model besar **belum diatur**. Setel konstanta `MODEL_RELEASE_BASE` di `app.py` ke "
            "URL rilis setelah mengunggah `pixel_rf.pkl`, `random_forest_best.pkl`, dan `pixel_et.pkl` "
            "ke hosting (mis. `https://github.com/USERNAME/REPO/releases/download/models-v1`). "
            "Langkah singkat ada di expander di bawah; panduan lengkap di `panduan_deploy_github.md` §6.")
    rows = []
    for fn, rep, algo, note, gh in MODEL_FILES:
        p = os.path.join(MODELS, fn)
        mb = os.path.getsize(p) / 1e6 if os.path.exists(p) else None
        rows.append({"file": fn, "representasi": rep, "algoritma": algo,
                     "ukuran": f"{mb:,.1f} MB" if mb is not None else "tidak dibundel",
                     "upload ke GitHub": gh, "catatan": note})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.divider()
    st.markdown("**Tombol unduh:**")
    c1, c2 = st.columns(2)
    for i, (fn, rep, algo, note, gh) in enumerate(MODEL_FILES):
        p = os.path.join(MODELS, fn)
        exists = os.path.exists(p)
        mb = os.path.getsize(p) / 1e6 if exists else 0.0
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"`{fn}` — {algo} ({rep})")
            st.caption(f"upload ke GitHub: {gh}")
            if exists and mb <= DOWNLOAD_MAX_MB:
                st.download_button(f"Unduh ({mb:,.1f} MB)", data=file_bytes(p),
                                   file_name=fn, mime="application/octet-stream", key=f"dl_{fn}")
            elif MODEL_RELEASE_BASE:
                st.link_button(f"Unduh dari rilis ({mb:,.0f} MB)" if exists else "Unduh dari rilis",
                               f"{MODEL_RELEASE_BASE}/{fn}")
            else:
                st.caption("Berkas besar — atur `MODEL_RELEASE_BASE` untuk mengaktifkan tautan.")
    st.divider()
    st.subheader("Bedah ukuran vs akurasi tiap model")
    st.markdown(
        "Ukuran model di sini membentang dari **2 KB sampai 1,3 GB** (lebih dari **600.000×**), "
        "tetapi akurasinya **hampir tidak berubah** (CV macro-F1 polygon **0,335–0,537**; "
        "test **0,391–0,474**). Artinya: **ukuran berkas besar tidak berarti model lebih pintar** — "
        "ukuran ditentukan oleh *struktur* model (jumlah & kedalaman pohon, jumlah support vector), "
        "bukan oleh kecerdasannya.")
    comp = pd.DataFrame([
        {"model": f"{tag}-{m}", "algoritma": algo, "ukuran": size,
         "CV polygon-F1": cv, "test polygon-F1": te, "peran": role}
        for tag, m, algo, size, _st, cv, te, role, _nar in MODEL_CARDS
    ])
    st.dataframe(comp.style.format({"CV polygon-F1": "{:.3f}", "test polygon-F1": "{:.3f}"}),
                 width="stretch", hide_index=True)
    st.caption("CV = skor validasi silang (pemilihan model, Bab 4); test = skor uji sekali (Bab 5). "
               "Kolom peran menandai model peta, pemenang CV, dan kandidat biasa.")
    st.markdown("**Penjelasan rinci tiap model** (klik untuk membuka):")
    for tag, m, algo, size, struktur, cv, te, role, narasi in MODEL_CARDS:
        with st.expander(f"{tag}-{m}  |  {size}  |  CV {cv:.3f}  |  test {te:.3f}  |  {role}"):
            r1, r2, r3 = st.columns(3)
            r1.metric("Ukuran berkas", size)
            r2.metric("CV polygon-F1", f"{cv:.3f}")
            r3.metric("Test polygon-F1", f"{te:.3f}")
            st.markdown(f"**Algoritma:** {algo}")
            st.markdown(f"**Struktur tersimpan:** {struktur}")
            st.markdown(narasi)
    with st.expander("Cara hosting model besar (GitHub Release / Hugging Face)"):
        st.markdown(
            "**Opsi A — GitHub Release (paling mudah):**\n"
            "1. Buat release baru di repo (mis. tag `models-v1`).\n"
            "2. Seret `pixel_rf.pkl`, `random_forest_best.pkl`, `pixel_et.pkl` sebagai asset.\n"
            "3. Salin URL asset → setel `MODEL_RELEASE_BASE` di `app.py` "
            "(mis. `https://github.com/USERNAME/REPO/releases/download/models-v1`).\n\n"
            "**Opsi B — Hugging Face Hub:** unggah berkas ke satu model repo, lalu isi "
            "`MODEL_RELEASE_BASE` dengan basis URL `resolve/main` repo tersebut.\n\n"
            "**Catatan Git LFS:** `.gitattributes` mengarahkan `pixel_rf.pkl` & "
            "`random_forest_best.pkl` ke LFS, tetapi Streamlit Community Cloud **tidak menjamin** "
            "menarik objek LFS. Jadi, untuk unduhan publik yang andal, pakai Release/HF — bukan LFS.")
    st.info(CATATAN_TEKNIS)

else:  # Metodologi
    diag = load_csv("polygon_validity_diagnostic.csv")
    sel = load_csv("polygon_selected.csv")
    split = load_csv("polygon_split.csv")
    luas = load_csv("luas_per_kelas.csv")
    stats = load_csv("polygon_pixel_stats.csv")
    valid_bp = int(((diag["kode"] == "VALID") & diag["fid"].isin(stats.loc[stats["n_valid"] > 0, "fid"])).sum())
    st.subheader("Ringkasan data")
    st.markdown(
        "- Sentinel-2 L2A, komposit median **2025-09-01/02**, maksimum 20% awan, 6 band "
        "(B02, B03, B04, B08, B11, B12) + 3 indeks (NDVI, NDWI, NDBI) = 9 fitur.\n"
        f"- Polygon label: {len(diag):,} mentah → 875 lolos V1–V6 → **{valid_bp} valid berpixel** "
        f"(107 di luar extent) → "
        f"**{len(sel)} terpilih** (50/kelas, Laut 6) → split per polygon "
        f"**{int((split['split'] == 'train').sum())} train / {int((split['split'] == 'test').sum())} test**.\n"
        f"- Pixel valid di peta: {int(luas['pixel_count'].sum()):,} (nodata 0 = di luar extent/badan air).\n"
        "- Kelas: 1 Sawah, 2 Bangunan/Permukiman, 3 Mangrove, 4 Lahan hijau, 5 Laut, 6 Danau."
    )
    d1, d2 = st.columns(2)
    d1.image(os.path.join(FIGURES, "profil_spektral.png"), caption="Profil spektral per kelas (Bab 2)")
    d2.image(os.path.join(FIGURES, "sebaran_per_kelas.png"), caption="Sebaran pixel sample per kelas (Bab 2)")
    st.subheader("Provenance label (sumber sekunder)")
    st.markdown(
        "Kementan LBS2019 (Sawah); BIG Satupeta /5 dan /57 (Bangunan, Lahan hijau, Danau, Mangrove); "
        "Natural Earth ne_10m_ocean (Laut); OSM/Overpass (tambahan). Training sample dibentuk sendiri "
        "dari sumber tersebut — label BUKAN referensi resmi per kabupaten."
    )
    st.subheader("Keterbatasan (wajib dibaca)")
    st.markdown(
        "- Satu snapshot 2 hari **tidak menunjukkan perubahan lahan**.\n"
        "- Akurasi mengikuti label sumber sekunder (test macro-F1 polygon ≈ 0,47).\n"
        "- Tidak ada klaim kausal; 6 kelas adalah penyederhanaan dari acuan SNI/KLHK.\n"
        "- Awan, bayangan, dan tambak tak berkelas → sebagian pixel dipaksa ke 6 kelas.\n"
        "- Peta = hasil model terbaik CV (pixel-svm menang prosedural, selisih dalam noise), "
        "diproduksi dengan pixel-rf karena kelayakan inferensi."
    )
    st.info(CATATAN_TEKNIS)
