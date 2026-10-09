"""Download & bangun data training polygon 6 kelas Jawa Timur.
Jalankan: python download_training_data.py
Output: training_samples_jawa_timur.gpkg / .geojson, training_samples_summary.csv,
        download_training_log.txt, boundary_jatim.geojson
"""
import io, json, logging, sys, time, zipfile
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

RANDOM_STATE = 42
TARGET_MIN, TARGET_MAX = 100, 500
MIN_AREA_M2 = 1000  # ~10 piksel Sentinel-2 10m
AREA_CRS = "EPSG:6933"  # equal-area untuk hitung luas
OUT_CRS = "EPSG:4326"

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "download_training_log.txt"
GPKG_FILE = BASE_DIR / "training_samples_jawa_timur.gpkg"
GEOJSON_FILE = BASE_DIR / "training_samples_jawa_timur.geojson"
CSV_FILE = BASE_DIR / "training_samples_summary.csv"
BOUNDARY_FILE = BASE_DIR / "boundary_jatim.geojson"

KEMENTAN = "https://sig02.pertanian.go.id/server/rest/services/Sawah/LBS2019/MapServer/22"
BIG3 = "https://kspservices.big.go.id/satupeta/rest/services/PUBLIK/SUMBER_DAYA_ALAM_DAN_LINGKUNGAN/MapServer/3"
BIG57 = "https://kspservices.big.go.id/satupeta/rest/services/PUBLIK/SUMBER_DAYA_ALAM_DAN_LINGKUNGAN/MapServer/57"
GADM_IDN1 = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_1.json"
NE_OCEAN_ZIP = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
CACHE_DIR = BASE_DIR / "_cache_dl"
# bbox Jatim (lon_min, lat_min, lon_max, lat_max) format ArcGIS envelope string
JATIM_BBOX = "110.89,-8.79,116.28,-5.04"
# Target unduhan per kategori BIG PT 10K (sebelum clip & sampling akhir)
TARGET_DL_PER_CAT = 800

# Kategori BIG Penggunaan Tanah 10K (layer 3, cakupan Jawa).
# CATATAN: BIG PT 50K (layer 5) TIDAK mencakup Jatim (investigasi: 34 ribu fitur
# nasional diunduh, 0 di Jatim; extent kategori di Kalimantan/Sulawesi/Sumatera).
CLASS2_CATS = [
    "Kampung Jarang Tidak Teratur", "Kampung Padat Tidak Teratur",
    "Kampung Jarang Teratur", "Kampung Padat Teratur",
    "Perumahan Padat", "Perumahan Jarang",
]
CLASS4_CATS = [
    "Kebun Campuran", "Tegalan/Ladang", "Semak", "Hutan Belukar",
    "Perkebunan Sudah Menghasilkan", "Kebun Buah-buahan", "Padang Rumput",
    "Hutan Sejenis", "Perkebunan Belum Menghasilkan", "Kebun Sayuran",
    "Sabana", "Alang \u2013 Alang", "Kebun Biofarmaka/Tumbuhan Obat",
    "Kebun Tanaman Hias", "Perkebunan", "Hutan Kota",
]
CLASS4_EXCLUDED_AMBIGUOUS = [
    # sawah -> kelas 1
    "Sawah Irigasi 2x Padi/thn", "Sawah Irigasi", "Sawah Irigasi Lebih Dari 2x Padi/thn",
    "Sawah Irigasi 1x Padi/thn", "Tadah Hujan", "Hujan Lebat",
    "2x Padi + Palawija/thn", "1x Padi/thn + Palawija/thn",
    "Pasang Surut 2x Padi/ th", "Pasang Surut 2x Padi + Palawija/ th",
    "Pasang Surut 1x Padi + Palawija/ th", "Pasang Surut 1x Padi/ th",
    # air/tambak -> bukan lahan hijau
    "Sungai/Danau", "Tambak", "Kolam Air Tawar", "Waduk", "Bencah", "Lebak",
    "Perairan Bekas Tambang", "Penggaraman",
    # terbangun/lainnya
    "Kampung Jarang Tidak Teratur", "Kampung Padat Tidak Teratur",
    "Aneka Industri", "Jasa Pendidikan", "Tanah Kosong Sudah Diperuntukan",
    "Tanah Terbuka Sementara (Land Clearing)", "Tanah Tandus", "Tanah Rusak",
    "Pasir", "Taman Umum", "Taman Private",
]
# Kelas 6: layer 3 tidak punya kategori danau alami; hanya Waduk (dicatat).
CLASS6_CATS = ["Waduk"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger()
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "opentraining-jatim/1.0"})

def http_get(url, params=None, timeout=120, retries=4):
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning(f"GET gagal ({i+1}/{retries}): {url[:120]} err={e}")
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"GET gagal setelah {retries}x: {url[:120]}")

def arcgis_count(base, where, geometry=None):
    p = {"where": where, "returnCountOnly": "true", "f": "json"}
    if geometry:
        p.update({"geometry": geometry, "geometryType": "esriGeometryEnvelope",
                  "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"})
    r = http_get(base + "/query", p)
    return int(r.json().get("count", 0))

def _to_2d(coords):
    # ArcGIS kadang mengembalikan koordinat 4D [x, y, z, m] dengan m=null.
    # Pangkas ke 2D agar shapely tidak error float(None).
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [coords[0], coords[1]]
    return [_to_2d(c) for c in coords]

def _strip_zm(feats):
    for f in feats:
        g = (f or {}).get("geometry") or {}
        if "coordinates" in g:
            try:
                g["coordinates"] = _to_2d(g["coordinates"])
            except Exception:
                f["geometry"] = None
    return [f for f in feats if (f or {}).get("geometry")]

def arcgis_fetch_geojson(base, where, out_fields="*", paginate=True, oid_field="objectid",
                           geometry=None, cache_name=None, expected=None):
    """Ambil semua fitur (f=geojson). Kembalikan list fitur geojson.
    geometry: bbox string 'xmin,ymin,xmax,ymax' (filter spasial server).
    cache_name: jika ada file _cache_dl/<name>.json, pakai itu (hemat waktu).
    expected: perkiraan jumlah (dari count) -> batas halaman pengaman."""
    if cache_name:
        CACHE_DIR.mkdir(exist_ok=True)
        cf = CACHE_DIR / f"{cache_name}.json"
        if cf.exists():
            try:
                feats = json.loads(cf.read_text(encoding="utf-8"))
                log.info(f"  cache HIT {cache_name}: {len(feats)} fitur")
                return feats
            except Exception as e:
                log.warning(f"  cache rusak {cache_name}: {e}, unduh ulang")
    feats = []
    if paginate:
        max_pages = (expected // 1000 + 3) if expected else 3000
        offset, pages = 0, 0
        while True:
            p = {"where": where, "outFields": out_fields, "f": "geojson",
                 "outSR": "4326", "returnGeometry": "true",
                 "orderByFields": oid_field,
                 "resultOffset": offset, "resultRecordCount": 1000}
            if geometry:
                p.update({"geometry": geometry, "geometryType": "esriGeometryEnvelope",
                          "inSR": "4326", "spatialRel": "esriSpatialRelIntersects"})
            r = http_get(base + "/query", p, timeout=180)
            fc = r.json()
            batch = fc.get("features", [])
            if not batch:
                break
            # pengaman: server yang mengabaikan offset akan mengulang halaman sama
            if feats and batch[0] == feats[-len(batch)] if len(feats) >= len(batch) else False:
                log.warning("  halaman berulang, stop")
                break
            feats.extend(batch)
            pages += 1
            log.info(f"  {base[-14:]} where=[{where[:60]}] offset={offset} dapat={len(batch)} total={len(feats)} exc={fc.get('exceededTransferLimit')}")
            if not fc.get("exceededTransferLimit") or pages >= max_pages:
                break
            offset += 1000
    else:  # tanpa pagination (Kementan): 1 query, harus < maxRecord
        p = {"where": where, "outFields": out_fields, "f": "geojson",
             "outSR": "4326", "returnGeometry": "true"}
        r = http_get(base + "/query", p, timeout=300)
        fc = r.json()
        feats = fc.get("features", [])
        log.info(f"  {base[-14:]} where=[{where[:60]}] dapat={len(feats)} exc={fc.get('exceededTransferLimit')}")
    feats = _strip_zm(feats)
    if cache_name:
        try:
            (CACHE_DIR / f"{cache_name}.json").write_text(json.dumps(feats), encoding="utf-8")
        except Exception as e:
            log.warning(f"  gagal tulis cache {cache_name}: {e}")
    return feats

def where_cat(field, value, mod=0, oid="objectid"):
    w = f"{field}='{value}'"
    if mod and mod > 1:
        w += f" AND MOD({oid}, {mod})=0"
    return w

def feats_to_gdf(feats):
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=OUT_CRS)
    return gpd.GeoDataFrame.from_features(feats, crs=OUT_CRS)

def clean_clip(gdf, aoi, label):
    n0 = len(gdf)
    if gdf.empty:
        log.info(f"{label}: kosong, skip")
        return gdf
    gdf = gdf.to_crs(OUT_CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gpd.clip(gdf, aoi)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    gdf = gdf.drop_duplicates(subset=["geometry"])
    if not gdf.empty:
        a = gdf.to_crs(AREA_CRS).geometry.area
        gdf = gdf[a.values >= MIN_AREA_M2]
    # explode multi agar tiap polygon satu baris
    if not gdf.empty:
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    log.info(f"{label}: {n0} -> setelah clip/clean: {len(gdf)}")
    return gdf

def finalize(gdf, class_id, class_name, source, src_field, aoi):
    gdf = clean_clip(gdf, aoi, f"Class {class_id} {class_name}")
    if gdf.empty:
        return gdf
    gdf["class_id"] = class_id
    gdf["class_name"] = class_name
    gdf["source"] = source
    gdf["source_class"] = gdf[src_field].astype(str) if src_field in gdf.columns else ""
    keep = ["geometry", "class_id", "class_name", "source", "source_class"]
    gdf = gdf[keep]
    if len(gdf) > TARGET_MAX:
        gdf = gdf.sample(n=TARGET_MAX, random_state=RANDOM_STATE).reset_index(drop=True)
        log.info(f"Class {class_id}: sampling ke {TARGET_MAX} (seed {RANDOM_STATE})")
    return gdf

def get_boundary():
    log.info("Download boundary GADM IDN level-1 ...")
    try:
        r = http_get(GADM_IDN1, timeout=180)
        gj = r.json()
        gdf = gpd.GeoDataFrame.from_features(gj["features"] if "features" in gj else gj, crs="EPSG:4326")
        cols = [c for c in ["NAME_1", "VARNAME_1", "HASC_1", "CC_1"] if c in gdf.columns]
        log.info(f"Kolom GADM: {list(gdf.columns)}")
        m = pd.Series(False, index=gdf.index)
        norm = lambda s: s.astype(str).str.lower().str.replace(r"\s+", "", regex=True)
        if "NAME_1" in gdf.columns:
            m |= norm(gdf["NAME_1"]).str.contains("jawatimur", na=False)
        if "VARNAME_1" in gdf.columns:
            m |= norm(gdf["VARNAME_1"]).str.contains("jawatimur|eastjava", na=False)
        if "HASC_1" in gdf.columns:
            m |= gdf["HASC_1"].astype(str).str.strip().eq("ID.JI")
        jatim = gdf[m]
        assert not jatim.empty, "Jawa Timur tidak ketemu di GADM"
        jatim = jatim.dissolve().geometry.to_frame("geometry")
        jatim = gpd.GeoDataFrame(jatim, crs="EPSG:4326").reset_index(drop=True)
        log.info(f"Boundary bounds: {jatim.total_bounds.tolist()}")
        assert jatim.total_bounds[2] > 112.0, f"Boundary salah provinsi? bounds={jatim.total_bounds.tolist()}"
        jatim.to_file(BOUNDARY_FILE, driver="GeoJSON")
        log.info(f"Boundary Jatim tersimpan: {BOUNDARY_FILE} (sumber: GADM v4.1 IDN_1)")
        return jatim
    except Exception as e:
        log.error(f"Boundary GADM gagal: {e}; fallback bbox kasar Jatim")
        aoi = gpd.GeoDataFrame(geometry=[box(110.8, -9.0, 114.6, -6.7)], crs="EPSG:4326")
        aoi.to_file(BOUNDARY_FILE, driver="GeoJSON")
        return aoi

def main():
    log.info("=== MULAI download_training_data.py ===")
    aoi = get_boundary()
    aoi_area_km2 = aoi.to_crs(AREA_CRS).geometry.area.sum() / 1e6
    log.info(f"Luas AOI Jatim: {aoi_area_km2:,.0f} km2")

    results, stats = {}, {}

    # --- Kelas 1 Sawah (Kementan, tanpa pagination -> MOD 100) ---
    try:
        w = "MOD(OBJECTID, 100)=0"
        c = arcgis_count(KEMENTAN, "1=1"); log.info(f"Sawah total di layer Jatim: {c}")
        c100 = arcgis_count(KEMENTAN, w); log.info(f"Sawah MOD100: {c100}")
        f = arcgis_fetch_geojson(KEMENTAN, w, out_fields="*", paginate=False, oid_field="OBJECTID",
                                   cache_name="c1_sawah")
        g = feats_to_gdf(f)
        log.info(f"Sawah kolom: {list(g.columns)}")
        src = "WADMKK_1" if "WADMKK_1" in g.columns else ("QNAME_19" if "QNAME_19" in g.columns else None)
        if src is None:
            g["src_tmp"] = "Sawah"; src = "src_tmp"
        results[1] = finalize(g, 1, "Sawah", "Kementan", src, aoi)
        stats["sawah_dl"] = len(f)
    except Exception as e:
        log.error(f"KELAS 1 GAGAL: {e}"); results[1] = gpd.GeoDataFrame(geometry=[], crs=OUT_CRS)

    # --- Kelas 3 Mangrove (BIG 57, prov filter) ---
    try:
        w = "prov='Jawa Timur'"
        c = arcgis_count(BIG57, w); log.info(f"Mangrove Jatim count: {c}")
        f = arcgis_fetch_geojson(BIG57, w, paginate=True, cache_name="c3_mangrove",
                                   expected=c)
        g = feats_to_gdf(f)
        log.info(f"Mangrove unique kttj: {g['kttj'].unique().tolist() if 'kttj' in g.columns else []}")
        results[3] = finalize(g, 3, "Mangrove", "BIG", "kttj" if "kttj" in g.columns else None, aoi)
        stats["mangrove_dl"] = len(f)
    except Exception as e:
        log.error(f"KELAS 3 GAGAL: {e}"); results[3] = gpd.GeoDataFrame(geometry=[], crs=OUT_CRS)

    # --- Kelas 2 / 4 / 6 dari BIG PT 10K (layer 3, filter spasial Jatim + MOD) ---
    for cid, cname, cats in [(2, "Bangunan / Permukiman", CLASS2_CATS),
                             (4, "Lahan hijau", CLASS4_CATS),
                             (6, "Danau", CLASS6_CATS)]:
        frames = []
        for cat in cats:
            safe = "".join(ch if ch.isalnum() else "_" for ch in cat)[:40]
            try:
                c = arcgis_count(BIG3, f"ptnobjname='{cat}'", geometry=JATIM_BBOX)
                log.info(f"C{cid} kat '{cat}' Jatim count={c}")
                if c == 0:
                    continue
                # Waduk (kelas 6): ambil semua; lainnya sampling server ~800/kat
                if cid == 6:
                    w, mod = f"ptnobjname='{cat}'", 0
                else:
                    mod = max(1, round(c / TARGET_DL_PER_CAT))
                    w = where_cat("ptnobjname", cat, mod)
                f = arcgis_fetch_geojson(BIG3, w, paginate=True, geometry=JATIM_BBOX,
                                         cache_name=f"c{cid}_{safe}_m{mod}", expected=c)
                g = feats_to_gdf(f)
                if not g.empty:
                    frames.append(g)
            except Exception as e:
                log.error(f"C{cid} kat '{cat}' GAGAL: {e}")
        g = pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame(geometry=[], crs=OUT_CRS)
        if not g.empty and not isinstance(g, gpd.GeoDataFrame):
            g = gpd.GeoDataFrame(g, crs=OUT_CRS)
        if not g.empty:
            uv = g["ptnobjname"].unique().tolist() if "ptnobjname" in g.columns else []
            log.info(f"C{cid} unique ptnobjname terdownload: {uv}")
        if cid == 6 and not g.empty:
            log.info("CATATAN KELAS 6: memakai Waduk sbg proksi danau "
                     "(layer 3 tak ada kategori danau alami; waduk != danau alami).")
        results[cid] = finalize(g, cid, cname, "BIG",
                                "ptnobjname" if (not g.empty and "ptnobjname" in g.columns) else None, aoi)
        stats[f"c{cid}_dl"] = sum(len(x) for x in frames)

    # --- Kelas 5 Laut (Natural Earth ocean + grid offshore) ---
    try:
        log.info("Download Natural Earth ocean ...")
        r = http_get(NE_OCEAN_ZIP, timeout=180)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        tmpd = BASE_DIR / "_tmp_ne"; tmpd.mkdir(exist_ok=True)
        z.extractall(tmpd)
        shp = next(tmpd.glob("**/*.shp"))
        ocean = gpd.read_file(shp).to_crs(OUT_CRS)
        landm = aoi.to_crs(AREA_CRS)
        buf30 = gpd.GeoDataFrame(geometry=landm.buffer(30000), crs=AREA_CRS)
        water = gpd.overlay(gpd.GeoDataFrame(geometry=ocean.to_crs(AREA_CRS).geometry),
                            buf30, how="intersection")
        coast1k = gpd.GeoDataFrame(geometry=landm.buffer(1000), crs=AREA_CRS)
        water_clean = gpd.overlay(water, coast1k, how="difference")
        wc = water_clean.dissolve()
        xmin, ymin, xmax, ymax = wc.total_bounds
        cell, cells = 5000, []
        for x in np.arange(xmin, xmax, cell):
            for y in np.arange(ymin, ymax, cell):
                cells.append(box(x, y, x + cell, y + cell))
        grid = gpd.GeoDataFrame(geometry=cells, crs=AREA_CRS)
        grid["c"] = grid.geometry.centroid
        inside = grid.set_geometry("c").within(wc.geometry.iloc[0])
        grid = grid[inside.values].set_geometry("geometry").reset_index(drop=True)
        log.info(f"Kandidat sel laut offshore (>1km pantai, 5x5km): {len(grid)}")
        if len(grid) > TARGET_MAX:
            grid = grid.sample(n=TARGET_MAX, random_state=RANDOM_STATE).reset_index(drop=True)
        grid = grid.to_crs(OUT_CRS)
        grid["class_id"] = 5; grid["class_name"] = "Laut"
        grid["source"] = "Natural Earth"; grid["source_class"] = "ocean"
        # cleaning ringan
        grid["geometry"] = grid.geometry.make_valid()
        grid = gpd.clip(grid, gpd.GeoDataFrame(
            geometry=[box(*(aoi.to_crs(AREA_CRS).total_bounds[:2] - np.array([40000, 40000])),
                           *(aoi.to_crs(AREA_CRS).total_bounds[2:] + np.array([40000, 40000])))],
            crs=AREA_CRS).to_crs(OUT_CRS))
        results[5] = grid[["geometry", "class_id", "class_name", "source", "source_class"]]
        stats["laut_cells"] = len(grid)
    except Exception as e:
        log.error(f"KELAS 5 GAGAL: {e}"); results[5] = gpd.GeoDataFrame(geometry=[], crs=OUT_CRS)

    # --- Gabung & export ---
    order = [(1, "Sawah"), (2, "Bangunan / Permukiman"), (3, "Mangrove"),
             (4, "Lahan hijau"), (5, "Laut"), (6, "Danau")]
    allg = []
    for cid, cname in order:
        g = results.get(cid)
        if g is not None and not g.empty:
            allg.append(g)
        else:
            log.warning(f"Class {cid} {cname}: KOSONG (SOURCE NOT AVAILABLE jika gagal total)")
    merged = gpd.GeoDataFrame(pd.concat(allg, ignore_index=True), crs=OUT_CRS) if allg else \
        gpd.GeoDataFrame(columns=["geometry", "class_id", "class_name", "source", "source_class"], crs=OUT_CRS)
    merged.insert(0, "fid", range(len(merged)))
    merged.to_file(GPKG_FILE, layer="training_samples", driver="GPKG")
    merged.to_file(GEOJSON_FILE, driver="GeoJSON")
    log.info(f"Tersimpan: {GPKG_FILE} ({len(merged)} fitur) & {GEOJSON_FILE}")

    # --- Summary CSV per (kelas, source_class) ---
    rows = []
    m2 = merged.to_crs(AREA_CRS).geometry.area.values if not merged.empty else []
    merged["_area"] = m2
    for (cid, sc), grp in (merged.groupby(["class_id", "source_class"]) if not merged.empty else []):
        cn = grp["class_name"].iloc[0]; so = grp["source"].iloc[0]
        rows.append(dict(class_id=cid, class_name=cn, source=so, source_class=sc,
                         polygon_count=len(grp), total_area=grp["_area"].sum(),
                         mean_area=grp["_area"].mean(), median_area=grp["_area"].median()))
    pd.DataFrame(rows, columns=["class_id", "class_name", "source", "source_class",
                                "polygon_count", "total_area", "mean_area", "median_area"]
                 ).to_csv(CSV_FILE, index=False)
    log.info(f"Summary CSV: {CSV_FILE}")

    # --- Tabel & validasi akhir ---
    print("\n=== TRAINING DATA SUMMARY ===")
    tot_n, tot_a = 0, 0.0
    for cid, cname in order:
        g = results.get(cid)
        n = 0 if g is None or g.empty else len(g)
        a = 0.0 if g is None or g.empty else g.to_crs(AREA_CRS).geometry.area.sum()
        med = 0.0 if g is None or g.empty else g.to_crs(AREA_CRS).geometry.area.median()
        print(f"Class {cid} - {cname}\nJumlah polygon: {n} | total luas: {a:,.0f} m2 | median: {med:,.0f} m2")
        tot_n += n; tot_a += a
    print(f"\nTOTAL POLYGON: {tot_n}\nTOTAL LUAS: {tot_a:,.0f} m2")
    print(f"\nclass_id | class_name | jumlah_polygon | total_area | median_area")
    for cid, cname in order:
        g = results.get(cid)
        n = 0 if g is None or g.empty else len(g)
        a = 0.0 if g is None or g.empty else g.to_crs(AREA_CRS).geometry.area.sum()
        med = 0.0 if g is None or g.empty else g.to_crs(AREA_CRS).geometry.area.median()
        print(f"{cid} | {cname} | {n} | {a:,.0f} | {med:,.0f}")
    print(f"\nLokasi:\n- script: {Path(__file__).resolve()}\n- gpkg: {GPKG_FILE}\n- geojson: {GEOJSON_FILE}\n- csv: {CSV_FILE}\n- log: {LOG_FILE}\n- boundary: {BOUNDARY_FILE}")
    log.info("=== SELESAI ===")

if __name__ == "__main__":
    main()
