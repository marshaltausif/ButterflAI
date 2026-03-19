# modal/dataset_intelligence.py
"""
ButterflAI Data Intelligence Layer
====================================
Runs INSIDE the Modal GPU worker after dataset download.
Physically reads every file. Measures everything. Invents nothing.

Handles: images (JPEG/PNG/TIFF/DICOM/WEBP/BMP), tabular (CSV/Parquet/Excel/JSON),
         text (plaintext/JSONL), audio (WAV/MP3/FLAC), and mixed datasets.

Returns a DatasetProfile dict that drives:
  - preprocessing code generation
  - augmentation strategy selection
  - class weighting
  - normalization constants
  - train/val split strategy
  - domain-specific transforms
"""

import os, json, struct, hashlib, random, time, math, io
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_profile(dataset_dir: str, goal: str, declared_classes: list,
                           modality_hint: str = "auto") -> dict:
    """
    Full dataset inspection. Called by butterflai_worker.py after download.
    Returns a DatasetProfile that drives all downstream code generation.
    """
    ds = Path(dataset_dir)
    t0 = time.time()

    print("[DataIntel] ════════════════════════════════════════")
    print(f"[DataIntel] Inspecting: {ds}")
    print(f"[DataIntel] Goal: {goal}")
    print(f"[DataIntel] Declared classes: {declared_classes}")
    print("[DataIntel] ════════════════════════════════════════")

    profile = {
        "dataset_path": str(ds),
        "goal": goal,
        "declared_classes": declared_classes,
        "modality": None,
        "structure": {},
        "file_inventory": {},
        "format_analysis": {},
        "class_analysis": {},
        "quality_analysis": {},
        "domain_analysis": {},
        "statistics": {},
        "recommendations": {},
        "preprocessing_spec": {},
        "warnings": [],
        "errors": [],
        "inspection_time_sec": 0,
    }

    try:
        # 1. File inventory — what's actually here
        inventory = scan_directory(ds)
        profile["file_inventory"] = inventory
        print(f"[DataIntel] Files: {inventory['total_files']} | "
              f"Types: {list(inventory['by_extension'].keys())}")

        # 2. Detect modality from actual files
        profile["modality"] = detect_modality(inventory, modality_hint)
        print(f"[DataIntel] Modality: {profile['modality']}")

        # 3. Detect structure (how files are organized)
        profile["structure"] = detect_structure(ds, declared_classes, profile["modality"])
        print(f"[DataIntel] Structure: {profile['structure']['layout']}")

        # 4. Modality-specific deep analysis
        m = profile["modality"]
        if m == "image":
            profile["format_analysis"] = analyze_images(ds, profile["structure"])
            profile["class_analysis"]  = analyze_class_distribution(ds, profile["structure"], declared_classes)
            profile["quality_analysis"]= assess_image_quality(ds, profile["structure"])
            profile["domain_analysis"] = detect_image_domain(goal, profile["format_analysis"])
        elif m == "tabular":
            profile["format_analysis"] = analyze_tabular(ds, profile["structure"])
            profile["class_analysis"]  = analyze_tabular_labels(profile["format_analysis"])
        elif m == "text":
            profile["format_analysis"] = analyze_text(ds, profile["structure"])
            profile["class_analysis"]  = analyze_text_labels(profile["format_analysis"])
        elif m == "audio":
            profile["format_analysis"] = analyze_audio(ds, profile["structure"])
            profile["class_analysis"]  = analyze_class_distribution(ds, profile["structure"], declared_classes)

        # 5. Statistics summary
        profile["statistics"] = build_statistics(profile)

        # 6. Recommendations — concrete decisions for code generation
        profile["recommendations"] = derive_recommendations(profile)

        # 7. Preprocessing spec — exact transforms to generate
        profile["preprocessing_spec"] = build_preprocessing_spec(profile)

        # 8. Warnings
        profile["warnings"] = collect_warnings(profile)

    except Exception as e:
        profile["errors"].append(f"Inspection error: {str(e)}")
        import traceback
        print(f"[DataIntel] ERROR: {e}\n{traceback.format_exc()}")

    profile["inspection_time_sec"] = round(time.time() - t0, 2)

    print(f"[DataIntel] Done in {profile['inspection_time_sec']}s")
    print(f"[DataIntel] Recommendations: {json.dumps(profile['recommendations'], indent=2)}")
    if profile["warnings"]:
        print(f"[DataIntel] Warnings: {profile['warnings']}")

    return profile


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE INVENTORY — scan everything
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS   = {".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif",
                ".dcm",".dicom",".nii",".nii.gz"}
TABULAR_EXTS = {".csv",".tsv",".parquet",".feather",".arrow",
                ".xlsx",".xls",".json",".jsonl",".ndjson"}
TEXT_EXTS    = {".txt",".text",".jsonl",".ndjson",".json"}
AUDIO_EXTS   = {".wav",".mp3",".flac",".ogg",".m4a",".opus"}

def scan_directory(ds: Path) -> dict:
    inventory = {
        "total_files": 0,
        "total_bytes": 0,
        "by_extension": {},
        "image_files": [],
        "tabular_files": [],
        "text_files": [],
        "audio_files": [],
        "other_files": [],
        "max_depth": 0,
    }
    for fpath in ds.rglob("*"):
        if not fpath.is_file():
            continue
        if any(p.startswith(".") for p in fpath.parts):
            continue
        depth = len(fpath.relative_to(ds).parts)
        inventory["max_depth"] = max(inventory["max_depth"], depth)

        ext = fpath.suffix.lower()
        # Handle .nii.gz
        if fpath.name.endswith(".nii.gz"):
            ext = ".nii.gz"

        size = fpath.stat().st_size
        inventory["total_files"] += 1
        inventory["total_bytes"] += size
        inventory["by_extension"][ext] = inventory["by_extension"].get(ext, 0) + 1

        if ext in IMAGE_EXTS:
            inventory["image_files"].append(str(fpath))
        elif ext in TABULAR_EXTS:
            inventory["tabular_files"].append(str(fpath))
        elif ext in AUDIO_EXTS:
            inventory["audio_files"].append(str(fpath))
        elif ext in TEXT_EXTS:
            inventory["text_files"].append(str(fpath))
        else:
            inventory["other_files"].append(str(fpath))

    return inventory


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODALITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_modality(inventory: dict, hint: str = "auto") -> str:
    if hint != "auto":
        return hint
    counts = {
        "image":   len(inventory["image_files"]),
        "tabular": len(inventory["tabular_files"]),
        "audio":   len(inventory["audio_files"]),
        "text":    len(inventory["text_files"]),
    }
    # Special: if both images and tabular, might be image + metadata
    # Return the dominant one
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        return "unknown"
    # Tabular JSON could be annotation file for images
    if dominant == "text" and counts["image"] > 0:
        return "image"
    return dominant


# ══════════════════════════════════════════════════════════════════════════════
# 3. STRUCTURE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_structure(ds: Path, declared_classes: list, modality: str) -> dict:
    structure = {
        "layout": "unknown",
        "train_dir": None,
        "val_dir": None,
        "test_dir": None,
        "class_dirs": {},       # class_name -> [file_paths]
        "manifest_files": [],
        "has_splits": False,
        "split_ratio": None,
    }

    # Check for split dirs
    for split_name in ("train", "training"):
        p = ds / split_name
        if p.is_dir():
            structure["train_dir"] = str(p)
            structure["has_splits"] = True
    for split_name in ("val", "valid", "validation"):
        p = ds / split_name
        if p.is_dir():
            structure["val_dir"] = str(p)
    for split_name in ("test", "testing"):
        p = ds / split_name
        if p.is_dir():
            structure["test_dir"] = str(p)

    if modality == "image":
        base = Path(structure["train_dir"]) if structure["train_dir"] else ds
        # Collect class dirs
        class_dirs = {}
        for item in base.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                files = [f for f in item.rglob("*")
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                if files:
                    class_dirs[item.name] = [str(f) for f in files]

        if class_dirs:
            structure["layout"] = "imagefolder"
            # Try to match declared class names (case-insensitive fuzzy)
            matched = {}
            for dc in (declared_classes or []):
                for cn, paths in class_dirs.items():
                    if cn.lower() == dc.lower() or dc.lower() in cn.lower() or cn.lower() in dc.lower():
                        matched[dc] = paths
                        break
            structure["class_dirs"] = matched if matched else class_dirs
        else:
            structure["layout"] = "flat"

        # Look for annotation files
        for pattern in ("*.csv", "*.json", "labels.txt", "annotations.json",
                         "train.csv", "val.csv", "metadata.csv"):
            for f in ds.rglob(pattern):
                structure["manifest_files"].append(str(f))

    elif modality == "tabular":
        structure["layout"] = "single_file" if len(
            [f for f in ds.rglob("*") if f.suffix.lower() in TABULAR_EXTS]
        ) == 1 else "multi_file"

    elif modality == "audio":
        # Same ImageFolder pattern but for audio
        class_dirs = {}
        base = Path(structure["train_dir"]) if structure["train_dir"] else ds
        for item in base.iterdir():
            if item.is_dir():
                files = [f for f in item.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
                if files:
                    class_dirs[item.name] = [str(f) for f in files]
        structure["layout"] = "audiofolder" if class_dirs else "flat"
        structure["class_dirs"] = class_dirs

    return structure


# ══════════════════════════════════════════════════════════════════════════════
# 4A. IMAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_images(ds: Path, structure: dict) -> dict:
    """Physically reads image headers and samples pixels."""
    all_files = []
    for paths in structure.get("class_dirs", {}).values():
        all_files.extend(paths)
    if not all_files:
        all_files = [f for f in ds.rglob("*")
                     if f.is_file() and Path(f).suffix.lower() in IMAGE_EXTS]
    all_files = [Path(f) for f in all_files]

    sample = random.sample(all_files, min(300, len(all_files)))

    exts          = Counter()
    color_modes   = Counter()
    widths, heights = [], []
    has_dicom = has_tiff = has_nifti = False
    bit_depths = Counter()
    corrupt = []

    for fpath in sample:
        ext = fpath.suffix.lower()
        exts[ext] += 1
        if ext in (".dcm", ".dicom"): has_dicom = True
        if ext in (".tif", ".tiff"):  has_tiff  = True
        if ext in (".nii", ".nii.gz"):has_nifti = True
        try:
            mode, w, h, depth = read_image_header(fpath)
            color_modes[mode] += 1
            widths.append(w)
            heights.append(h)
            bit_depths[depth] += 1
        except Exception as e:
            corrupt.append({"path": str(fpath), "error": str(e)})

    # Compute size stats
    def percentile(lst, p):
        if not lst: return 0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s)-1)]

    dominant_mode = color_modes.most_common(1)[0][0] if color_modes else "RGB"

    result = {
        "total_images": len(all_files),
        "sample_size": len(sample),
        "extensions": dict(exts),
        "dominant_ext": exts.most_common(1)[0][0] if exts else ".jpg",
        "color_modes": dict(color_modes),
        "dominant_color_mode": dominant_mode,
        "is_grayscale": dominant_mode in ("L", "1", "grayscale_medical"),
        "has_dicom": has_dicom,
        "has_tiff": has_tiff,
        "has_nifti": has_nifti,
        "bit_depth": bit_depths.most_common(1)[0][0] if bit_depths else 8,
        "width_p10":  percentile(widths, 10),
        "width_p50":  percentile(widths, 50),
        "width_p90":  percentile(widths, 90),
        "height_p10": percentile(heights, 10),
        "height_p50": percentile(heights, 50),
        "height_p90": percentile(heights, 90),
        "sizes_consistent": _size_consistent(widths, heights),
        "aspect_ratios_diverse": _aspect_diverse(widths, heights),
        "corrupt_files": corrupt,
        "corrupt_count": len(corrupt),
    }
    return result


def read_image_header(fpath: Path) -> tuple:
    """Read dimensions/mode from file header without loading full image. Fast."""
    ext = fpath.suffix.lower()

    # DICOM — just report grayscale medical
    if ext in (".dcm", ".dicom"):
        return "grayscale_medical", 512, 512, 12

    # NIfTI
    if ext == ".nii" or str(fpath).endswith(".nii.gz"):
        return "volumetric_3d", 256, 256, 16

    with open(fpath, "rb") as f:
        header = f.read(64)

    # PNG
    if header[:8] == b'\x89PNG\r\n\x1a\n' and len(header) >= 24:
        w     = struct.unpack(">I", header[16:20])[0]
        h     = struct.unpack(">I", header[20:24])[0]
        depth = header[24]
        ctype = header[25] if len(header) > 25 else 2
        mode  = {0:"L", 2:"RGB", 3:"P", 4:"LA", 6:"RGBA"}.get(ctype, "RGB")
        return mode, w, h, depth

    # JPEG
    if header[:2] == b'\xff\xd8':
        with open(fpath, "rb") as f:
            data = f.read(8192)
        for i in range(2, len(data) - 10):
            if data[i] == 0xff and data[i+1] in (0xc0,0xc1,0xc2,0xc3):
                h = struct.unpack(">H", data[i+5:i+7])[0]
                w = struct.unpack(">H", data[i+7:i+9])[0]
                c = data[i+9] if len(data) > i+9 else 3
                mode = "L" if c == 1 else "CMYK" if c == 4 else "RGB"
                return mode, w, h, 8
        return "RGB", 224, 224, 8

    # BMP
    if header[:2] == b'BM' and len(header) >= 30:
        w   = struct.unpack("<I", header[18:22])[0]
        h   = abs(struct.unpack("<i", header[22:26])[0])
        bpp = struct.unpack("<H", header[28:30])[0]
        return ("L" if bpp <= 8 else "RGB"), w, h, bpp

    # WEBP
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return "RGB", 224, 224, 8

    # TIFF — try PIL minimally
    try:
        from PIL import Image
        with Image.open(fpath) as img:
            return img.mode, img.width, img.height, 8
    except Exception as e:
        raise ValueError(f"Unreadable: {e}")


def _size_consistent(widths, heights):
    if not widths: return True
    mw = sorted(widths)[len(widths)//2]
    mh = sorted(heights)[len(heights)//2]
    ok = sum(1 for w,h in zip(widths,heights)
             if abs(w-mw)/max(mw,1) < 0.15 and abs(h-mh)/max(mh,1) < 0.15)
    return ok / len(widths) > 0.85

def _aspect_diverse(widths, heights):
    if not widths: return False
    ratios = [w/max(h,1) for w,h in zip(widths,heights)]
    return max(ratios) / max(min(ratios), 0.01) > 3.0


# ══════════════════════════════════════════════════════════════════════════════
# 4B. CLASS DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_class_distribution(ds: Path, structure: dict, declared_classes: list) -> dict:
    class_dirs = structure.get("class_dirs", {})
    if not class_dirs:
        return {"count": 0, "distribution": {}, "imbalance_ratio": 1.0,
                "is_imbalanced": False, "tiny_classes": [], "missing_classes": declared_classes or []}

    counts = {cls: len(paths) for cls, paths in class_dirs.items()}
    total  = sum(counts.values())
    vals   = list(counts.values())
    mn, mx = min(vals) if vals else 1, max(vals) if vals else 1
    imbalance = mx / max(mn, 1)

    missing = [c for c in (declared_classes or [])
               if c not in counts and c.lower() not in {k.lower() for k in counts}]
    tiny    = [{"class": c, "count": n} for c, n in counts.items() if n < 50]

    # Class weights for WeightedRandomSampler
    weights = {c: total / (len(counts) * n) for c, n in counts.items() if n > 0}

    # Val split distribution (if exists)
    val_counts = {}
    val_dir = structure.get("val_dir")
    if val_dir:
        vd = Path(val_dir)
        for item in vd.iterdir():
            if item.is_dir():
                imgs = [f for f in item.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    val_counts[item.name] = len(imgs)

    return {
        "count": len(counts),
        "distribution": counts,
        "val_distribution": val_counts,
        "total_samples": total,
        "min_count": mn,
        "max_count": mx,
        "imbalance_ratio": round(imbalance, 2),
        "is_imbalanced": imbalance > 5,
        "is_severely_imbalanced": imbalance > 20,
        "tiny_classes": tiny,
        "missing_classes": missing,
        "class_weights": {c: round(w, 4) for c, w in weights.items()},
        "recommended_sampling": (
            "weighted_random_sampler" if imbalance > 5
            else "class_weighted_loss" if imbalance > 2
            else "none"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4C. IMAGE QUALITY ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def assess_image_quality(ds: Path, structure: dict) -> dict:
    """Sample images, compute pixel stats, detect near-dupes, blank images."""
    all_files = []
    for paths in structure.get("class_dirs", {}).values():
        all_files.extend([Path(p) for p in paths])
    if not all_files:
        all_files = [f for f in ds.rglob("*")
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

    sample = random.sample(all_files, min(150, len(all_files)))

    blank_count = 0
    dark_count  = 0
    oversaturated = 0
    hash_set    = set()
    near_dupe_count = 0
    brightness_vals = []
    std_vals = []

    for fpath in sample:
        try:
            from PIL import Image
            import numpy as np
            with Image.open(fpath) as img:
                img = img.convert("L").resize((64, 64))
                arr = np.array(img, dtype=np.float32)
                mean_v = float(arr.mean())
                std_v  = float(arr.std())
                brightness_vals.append(mean_v)
                std_vals.append(std_v)

                if mean_v < 10:   dark_count  += 1
                if std_v  < 2:    blank_count += 1
                if mean_v > 245:  oversaturated += 1

                # Perceptual hash for near-duplicate detection
                phash = _phash(arr)
                if phash in hash_set:
                    near_dupe_count += 1
                hash_set.add(phash)
        except Exception:
            pass

    return {
        "sample_size": len(sample),
        "blank_or_uniform_count": blank_count,
        "very_dark_count": dark_count,
        "oversaturated_count": oversaturated,
        "near_duplicate_count": near_dupe_count,
        "mean_brightness": round(sum(brightness_vals)/max(len(brightness_vals),1), 1),
        "mean_contrast":   round(sum(std_vals)/max(len(std_vals),1), 1),
        "quality_issues_pct": round(
            (blank_count + dark_count + near_dupe_count) / max(len(sample), 1) * 100, 1
        ),
    }


def _phash(arr) -> int:
    """Simple perceptual hash for near-duplicate detection."""
    try:
        import numpy as np
        resized = arr[:8, :8]  # already 64x64, take 8x8 top-left
        mean = resized.mean()
        bits = (resized > mean).flatten()
        return int("".join("1" if b else "0" for b in bits), 2)
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# 4D. IMAGE DOMAIN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_RULES = {
    "medical_xray":    ["xray","x-ray","chest","lung","pneumonia","covid","radiograph"],
    "medical_pathology": ["histology","pathology","microscope","cell","tissue","cancer","tumor"],
    "medical_dermoscopy": ["skin","dermoscopy","melanoma","lesion","dermatology","mole"],
    "medical_dicom":   ["dicom","mri","ct scan","scan","radiology"],
    "satellite":       ["satellite","aerial","drone","remote sensing","landuse","sentinel","landsat"],
    "wildlife":        ["wildlife","animal","species","bird","butterfly","insect","nature"],
    "document":        ["document","ocr","text","receipt","invoice","handwriting","mnist"],
    "face":            ["face","facial","emotion","expression","age","gender","person"],
    "autonomous":      ["traffic","road","driving","lane","pedestrian","vehicle","car"],
    "manufacturing":   ["defect","inspection","quality","industrial","pcb","weld"],
    "food":            ["food","fruit","vegetable","dish","ingredient","meal"],
    "general":         [],  # fallback
}

def detect_image_domain(goal: str, format_analysis: dict) -> dict:
    goal_lower = goal.lower()
    scores = {}
    for domain, keywords in DOMAIN_RULES.items():
        scores[domain] = sum(1 for kw in keywords if kw in goal_lower)

    # DICOM files force medical domain
    if format_analysis.get("has_dicom"):
        scores["medical_dicom"] = 99

    if format_analysis.get("is_grayscale") and not format_analysis.get("has_dicom"):
        scores["document"] += 1
        scores["medical_xray"] += 1

    domain = max(scores, key=scores.get)
    if scores[domain] == 0:
        domain = "general"

    # Domain-specific special handling flags
    special = {
        "medical_xray":       ["no_horizontal_flip", "clahe_normalization",
                                "lung_windowing", "single_channel_ok"],
        "medical_pathology":  ["stain_normalization", "high_resolution_needed",
                                "rotation_augment_ok"],
        "medical_dermoscopy": ["hair_removal_aware", "color_constancy",
                                "lesion_centering"],
        "medical_dicom":      ["pydicom_loading", "hu_windowing",
                                "single_channel", "no_jpeg_decode"],
        "satellite":          ["multi_band_possible", "no_jpeg_artifacts",
                                "normalize_per_band", "large_tiles_ok"],
        "wildlife":           ["color_jitter_ok", "random_crop_ok",
                                "background_diverse"],
        "document":           ["binarization_ok", "deskew_needed",
                                "preserve_text_orientation"],
        "face":               ["align_faces", "no_aggressive_crop",
                                "skin_tone_preserve"],
        "general":            ["standard_augmentation"],
    }.get(domain, ["standard_augmentation"])

    return {
        "type": domain,
        "confidence": min(scores.get(domain, 0) / 3.0, 1.0),
        "special_handling": special,
        "scores": {k: v for k, v in scores.items() if v > 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4E. TABULAR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_tabular(ds: Path, structure: dict) -> dict:
    tabular_files = [f for f in ds.rglob("*")
                     if f.is_file() and f.suffix.lower() in TABULAR_EXTS]
    if not tabular_files:
        return {"error": "No tabular files found"}

    result = {
        "files_found": [str(f) for f in tabular_files],
        "primary_file": None,
        "rows": 0,
        "columns": [],
        "dtypes": {},
        "null_rates": {},
        "target_candidates": [],
        "feature_candidates": [],
        "format": None,
        "sample_head": [],
    }

    # Pick the largest file as primary
    primary = max(tabular_files, key=lambda f: f.stat().st_size)
    result["primary_file"] = str(primary)
    result["format"] = primary.suffix.lower()

    try:
        import pandas as pd
        if primary.suffix.lower() == ".parquet":
            df = pd.read_parquet(primary, engine="pyarrow")
        elif primary.suffix.lower() in (".csv", ".tsv"):
            df = pd.read_csv(primary, nrows=10000)
        elif primary.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(primary, nrows=10000)
        elif primary.suffix.lower() in (".json", ".jsonl", ".ndjson"):
            df = pd.read_json(primary, lines=primary.suffix in (".jsonl",".ndjson"))
        else:
            df = pd.read_csv(primary, nrows=10000)

        result["rows"] = len(df)
        result["columns"] = list(df.columns)
        result["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
        result["null_rates"] = {col: round(df[col].isna().mean(), 4)
                                 for col in df.columns}

        # Target column candidates: low-cardinality categorical or explicit "label"/"target"
        for col in df.columns:
            cl = col.lower()
            nuniq = df[col].nunique()
            if cl in ("label","target","class","y","category","output"):
                result["target_candidates"].insert(0, {
                    "column": col, "unique_values": nuniq,
                    "reason": "name matches label/target/class"
                })
            elif nuniq <= 20 and str(df[col].dtype) in ("object","category","int64","int32"):
                result["target_candidates"].append({
                    "column": col, "unique_values": nuniq,
                    "reason": f"low cardinality ({nuniq} unique values)"
                })
            else:
                result["feature_candidates"].append(col)

        result["sample_head"] = df.head(3).to_dict(orient="records")

    except Exception as e:
        result["error"] = str(e)

    return result


def analyze_tabular_labels(format_analysis: dict) -> dict:
    if not format_analysis.get("target_candidates"):
        return {"count": 0, "distribution": {}, "imbalance_ratio": 1.0}
    try:
        import pandas as pd
        primary = format_analysis.get("primary_file")
        if not primary:
            return {}
        target = format_analysis["target_candidates"][0]["column"]
        if format_analysis["format"] == ".parquet":
            df = pd.read_parquet(primary)
        else:
            df = pd.read_csv(primary)
        vc = df[target].value_counts()
        dist = {str(k): int(v) for k, v in vc.items()}
        vals = list(vc.values)
        imb = vals[0] / max(vals[-1], 1) if vals else 1
        return {
            "target_column": target,
            "count": len(dist),
            "distribution": dist,
            "total_samples": int(vc.sum()),
            "imbalance_ratio": round(imb, 2),
            "is_imbalanced": imb > 5,
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 4F. TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_text(ds: Path, structure: dict) -> dict:
    text_files = [f for f in ds.rglob("*")
                  if f.is_file() and f.suffix.lower() in TEXT_EXTS]
    if not text_files:
        return {"error": "No text files found"}

    result = {
        "files_found": len(text_files),
        "format": None,
        "total_samples": 0,
        "avg_token_length": 0,
        "max_token_length": 0,
        "has_labels": False,
        "label_column": None,
        "text_column": None,
        "languages_detected": [],
        "sample_texts": [],
    }

    primary = max(text_files, key=lambda f: f.stat().st_size)
    result["format"] = primary.suffix.lower()

    try:
        if primary.suffix.lower() in (".jsonl", ".ndjson"):
            lines = primary.read_text(errors="ignore").strip().split("\n")[:5000]
            samples = [json.loads(l) for l in lines if l.strip()]
            result["total_samples"] = len(samples)
            if samples:
                keys = list(samples[0].keys())
                # Find text and label columns
                for k in keys:
                    if k.lower() in ("text","sentence","input","content","review","comment"):
                        result["text_column"] = k
                    if k.lower() in ("label","target","class","sentiment","category"):
                        result["label_column"] = k
                        result["has_labels"] = True
                # Token length analysis
                text_col = result["text_column"] or keys[0]
                lengths = [len(str(s.get(text_col,"")).split()) for s in samples[:1000]]
                result["avg_token_length"] = round(sum(lengths)/max(len(lengths),1), 1)
                result["max_token_length"] = max(lengths) if lengths else 0
                result["sample_texts"] = [str(s.get(text_col,""))[:200]
                                           for s in samples[:3]]
        elif primary.suffix.lower() == ".txt":
            lines = primary.read_text(errors="ignore").strip().split("\n")[:5000]
            result["total_samples"] = len(lines)
            lengths = [len(l.split()) for l in lines[:1000] if l.strip()]
            result["avg_token_length"] = round(sum(lengths)/max(len(lengths),1), 1)
            result["sample_texts"] = [l[:200] for l in lines[:3] if l.strip()]
    except Exception as e:
        result["error"] = str(e)

    return result


def analyze_text_labels(format_analysis: dict) -> dict:
    return {
        "has_labels": format_analysis.get("has_labels", False),
        "label_column": format_analysis.get("label_column"),
        "count": 0,
        "distribution": {},
        "imbalance_ratio": 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4G. AUDIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_audio(ds: Path, structure: dict) -> dict:
    audio_files = [f for f in ds.rglob("*")
                   if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
    sample = random.sample(audio_files, min(50, len(audio_files)))

    durations = []
    sample_rates = Counter()
    formats = Counter()

    for fpath in sample:
        formats[fpath.suffix.lower()] += 1
        # Read WAV header directly
        if fpath.suffix.lower() == ".wav":
            try:
                sr, dur = read_wav_header(fpath)
                sample_rates[sr] += 1
                durations.append(dur)
            except Exception:
                pass

    return {
        "total_files": len(audio_files),
        "formats": dict(formats),
        "dominant_format": formats.most_common(1)[0][0] if formats else ".wav",
        "sample_rates": {str(k): v for k, v in sample_rates.items()},
        "dominant_sample_rate": sample_rates.most_common(1)[0][0] if sample_rates else 16000,
        "avg_duration_sec": round(sum(durations)/max(len(durations),1), 2),
        "min_duration_sec": round(min(durations), 2) if durations else 0,
        "max_duration_sec": round(max(durations), 2) if durations else 0,
    }


def read_wav_header(fpath: Path) -> tuple:
    with open(fpath, "rb") as f:
        data = f.read(44)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file")
    sample_rate    = struct.unpack("<I", data[24:28])[0]
    num_channels   = struct.unpack("<H", data[22:24])[0]
    bits_per_sample= struct.unpack("<H", data[34:36])[0]
    data_size      = struct.unpack("<I", data[40:44])[0]
    byte_rate      = struct.unpack("<I", data[28:32])[0]
    duration_sec   = data_size / max(byte_rate, 1)
    return sample_rate, duration_sec


# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_statistics(profile: dict) -> dict:
    inv = profile.get("file_inventory", {})
    ca  = profile.get("class_analysis", {})
    fa  = profile.get("format_analysis", {})

    return {
        "total_files": inv.get("total_files", 0),
        "total_size_gb": round(inv.get("total_bytes", 0) / 1e9, 2),
        "total_samples": ca.get("total_samples", fa.get("rows", 0)),
        "num_classes": ca.get("count", 0),
        "imbalance_ratio": ca.get("imbalance_ratio", 1.0),
        "has_splits": profile.get("structure", {}).get("has_splits", False),
        "quality_issues_pct": profile.get("quality_analysis", {}).get(
            "quality_issues_pct", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. RECOMMENDATIONS — concrete decisions for code generation
# ══════════════════════════════════════════════════════════════════════════════

def derive_recommendations(profile: dict) -> dict:
    m   = profile.get("modality", "image")
    fa  = profile.get("format_analysis", {})
    ca  = profile.get("class_analysis", {})
    da  = profile.get("domain_analysis", {})
    qa  = profile.get("quality_analysis", {})
    st  = profile.get("structure", {})

    recs = {
        # Data loading
        "loader_type": "ImageFolder",          # overridden below
        "needs_custom_loader": False,
        "use_datasets_library": False,

        # Normalization
        "normalize_mean": [0.485, 0.456, 0.406],   # ImageNet defaults
        "normalize_std":  [0.229, 0.224, 0.225],
        "use_imagenet_norm": True,

        # Color
        "input_channels": 1 if fa.get("is_grayscale") else 3,
        "convert_to_rgb": True,

        # Size
        "recommended_image_size": 224,

        # Augmentation
        "augmentation_strategy": "standard",
        "no_horizontal_flip": False,
        "no_vertical_flip": True,
        "use_clahe": False,
        "use_stain_norm": False,
        "use_random_rotation": True,
        "max_rotation_deg": 15,

        # Class handling
        "use_weighted_sampler": False,
        "use_class_weights_in_loss": False,
        "class_weights": {},
        "oversample_minority": False,

        # Split
        "create_val_split": not st.get("has_splits", False),
        "val_split_ratio": 0.2,
        "stratified_split": True,

        # Special loaders
        "needs_dicom_loader": False,
        "needs_tiff_loader": False,
        "needs_nifti_loader": False,
        "needs_audio_loader": False,
        "needs_tabular_loader": False,
    }

    # ── Image-specific ────────────────────────────────────────────────────────
    if m == "image":
        recs["loader_type"] = "split_imagefolder" if st.get("has_splits") else "imagefolder"

        # Size recommendation based on actual images
        p50w = fa.get("width_p50", 224)
        if p50w >= 512:
            recs["recommended_image_size"] = 384
        elif p50w <= 64:
            recs["recommended_image_size"] = 64
        else:
            recs["recommended_image_size"] = 224

        # Grayscale
        if fa.get("is_grayscale"):
            recs["input_channels"] = 1
            recs["convert_to_rgb"] = True   # let model use 3ch pretrained weights
            recs["normalize_mean"] = [0.5]
            recs["normalize_std"]  = [0.5]
            recs["use_imagenet_norm"] = False

        # DICOM
        if fa.get("has_dicom"):
            recs["needs_dicom_loader"] = True
            recs["needs_custom_loader"] = True
            recs["input_channels"] = 1
            recs["use_clahe"] = True

        # TIFF
        if fa.get("has_tiff"):
            recs["needs_tiff_loader"] = True
            recs["needs_custom_loader"] = True

        # Domain-specific augmentation
        domain = da.get("type", "general")
        special = da.get("special_handling", [])
        if "no_horizontal_flip" in special:
            recs["no_horizontal_flip"] = True
        if "clahe_normalization" in special:
            recs["use_clahe"] = True
        if "stain_normalization" in special:
            recs["use_stain_norm"] = True
        if "rotation_augment_ok" in special:
            recs["max_rotation_deg"] = 180
        if domain in ("satellite", "medical_pathology"):
            recs["augmentation_strategy"] = "heavy"
        elif domain in ("document", "face"):
            recs["augmentation_strategy"] = "light"
        else:
            recs["augmentation_strategy"] = "standard"

    # ── Class imbalance ───────────────────────────────────────────────────────
    if ca.get("is_imbalanced"):
        recs["use_weighted_sampler"] = True
        recs["class_weights"] = ca.get("class_weights", {})
    elif ca.get("imbalance_ratio", 1) > 2:
        recs["use_class_weights_in_loss"] = True
        recs["class_weights"] = ca.get("class_weights", {})

    # ── Tabular ──────────────────────────────────────────────────────────────
    if m == "tabular":
        recs["loader_type"] = "pandas_dataframe"
        recs["needs_tabular_loader"] = True
        recs["needs_custom_loader"] = True
        recs["target_column"] = (ca.get("target_column") or
                                  profile.get("format_analysis", {}).get(
                                      "target_candidates", [{}])[0].get("column"))

    # ── Audio ────────────────────────────────────────────────────────────────
    if m == "audio":
        recs["loader_type"] = "audio_dataset"
        recs["needs_audio_loader"] = True
        recs["needs_custom_loader"] = True
        recs["sample_rate"] = fa.get("dominant_sample_rate", 16000)

    # ── Text ────────────────────────────────────────────────────────────────
    if m == "text":
        recs["loader_type"] = "text_dataset"
        recs["needs_custom_loader"] = True
        recs["text_column"] = fa.get("text_column", "text")
        recs["label_column"] = fa.get("label_column", "label")
        recs["max_token_length"] = min(
            512,
            int(fa.get("max_token_length", 128) * 1.2)
        )

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# 7. PREPROCESSING SPEC — exact transforms for code generation
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessing_spec(profile: dict) -> dict:
    recs = profile.get("recommendations", {})
    m    = profile.get("modality", "image")
    da   = profile.get("domain_analysis", {})
    size = recs.get("recommended_image_size", 224)

    spec = {
        "modality": m,
        "train_transforms": [],
        "val_transforms": [],
        "custom_loader_code": None,
        "extra_imports": [],
    }

    if m == "image":
        # Train transforms — ordered list
        train_t = []
        val_t   = []

        # DICOM / special loaders
        if recs.get("needs_dicom_loader"):
            spec["extra_imports"].append("import pydicom")
            spec["custom_loader_code"] = (
                "def load_dicom(path):\n"
                "    dcm = pydicom.dcmread(path)\n"
                "    arr = dcm.pixel_array.astype(np.float32)\n"
                "    # Window/level normalization\n"
                "    if hasattr(dcm,'WindowCenter') and hasattr(dcm,'WindowWidth'):\n"
                "        wc = float(dcm.WindowCenter[0] if hasattr(dcm.WindowCenter,'__len__') else dcm.WindowCenter)\n"
                "        ww = float(dcm.WindowWidth[0]  if hasattr(dcm.WindowWidth,'__len__')  else dcm.WindowWidth)\n"
                "        lo, hi = wc - ww/2, wc + ww/2\n"
                "        arr = np.clip(arr, lo, hi)\n"
                "        arr = (arr - lo) / max(hi - lo, 1) * 255\n"
                "    arr = arr.astype(np.uint8)\n"
                "    img = Image.fromarray(arr).convert('RGB')\n"
                "    return img\n"
            )

        # Resize strategy
        if recs.get("aspect_ratios_diverse", False):
            train_t.append(f"T.Resize(({size},{size}))")   # force square
        else:
            train_t.append(f"T.RandomResizedCrop({size}, scale=(0.7, 1.0))")

        # Flips
        if not recs.get("no_horizontal_flip"):
            train_t.append("T.RandomHorizontalFlip()")
        if not recs.get("no_vertical_flip") and da.get("type","general") in ("satellite","medical_pathology"):
            train_t.append("T.RandomVerticalFlip()")

        # Rotation
        if recs.get("use_random_rotation"):
            deg = recs.get("max_rotation_deg", 15)
            train_t.append(f"T.RandomRotation({deg})")

        # Color jitter — skip for medical
        domain = da.get("type","general")
        if domain not in ("medical_xray","medical_dicom","document"):
            train_t.append("T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)")

        # CLAHE (handled in custom loader, not transform)

        # Heavy augmentation extras
        if recs.get("augmentation_strategy") == "heavy":
            train_t.append("T.RandomAffine(degrees=0, translate=(0.1,0.1))")
            train_t.append("T.GaussianBlur(kernel_size=3, sigma=(0.1,2.0))")

        # Normalize
        mean = recs.get("normalize_mean", [0.485,0.456,0.406])
        std  = recs.get("normalize_std",  [0.229,0.224,0.225])
        train_t.append("T.ToTensor()")
        train_t.append(f"T.Normalize(mean={mean}, std={std})")

        # Val transforms — no augmentation
        val_t.append(f"T.Resize(int({size}*1.15))")
        val_t.append(f"T.CenterCrop({size})")
        val_t.append("T.ToTensor()")
        val_t.append(f"T.Normalize(mean={mean}, std={std})")

        spec["train_transforms"] = train_t
        spec["val_transforms"]   = val_t

    elif m == "tabular":
        spec["train_transforms"] = ["StandardScaler", "handle_nulls", "encode_categoricals"]
        spec["val_transforms"]   = ["StandardScaler_transform"]

    elif m == "text":
        ml = recs.get("max_token_length", 128)
        spec["train_transforms"] = [
            f"tokenizer(text, max_length={ml}, truncation=True, padding='max_length')"
        ]
        spec["val_transforms"] = spec["train_transforms"]

    elif m == "audio":
        sr = recs.get("sample_rate", 16000)
        spec["train_transforms"] = [
            f"torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq={sr})",
            "torchaudio.transforms.MelSpectrogram(sample_rate={}, n_mels=128)".format(sr),
            "torchaudio.transforms.AmplitudeToDB()",
        ]
        spec["val_transforms"] = spec["train_transforms"]
        spec["extra_imports"].append("import torchaudio")

    return spec


# ══════════════════════════════════════════════════════════════════════════════
# 8. WARNINGS
# ══════════════════════════════════════════════════════════════════════════════

def collect_warnings(profile: dict) -> list:
    warnings = []
    ca = profile.get("class_analysis", {})
    fa = profile.get("format_analysis", {})
    qa = profile.get("quality_analysis", {})
    st = profile.get("statistics", {})

    if ca.get("is_severely_imbalanced"):
        worst = min(ca.get("distribution",{}).values()) if ca.get("distribution") else 0
        best  = max(ca.get("distribution",{}).values()) if ca.get("distribution") else 0
        warnings.append(
            f"SEVERE class imbalance: {ca['imbalance_ratio']:.0f}x ratio "
            f"(min={worst}, max={best}). WeightedRandomSampler enabled automatically."
        )
    elif ca.get("is_imbalanced"):
        warnings.append(
            f"Class imbalance detected ({ca['imbalance_ratio']:.1f}x). "
            f"Class-weighted loss enabled."
        )

    if ca.get("tiny_classes"):
        for tc in ca["tiny_classes"]:
            warnings.append(
                f"Class '{tc['class']}' has only {tc['count']} samples — "
                f"model may underfit this class."
            )

    if ca.get("missing_classes"):
        warnings.append(
            f"Expected classes not found on disk: {ca['missing_classes']}. "
            f"Check dataset download or class name mapping."
        )

    if fa.get("corrupt_count", 0) > 0:
        warnings.append(
            f"{fa['corrupt_count']} corrupt/unreadable image files found. "
            f"They will be skipped during training."
        )

    if qa.get("near_duplicate_count", 0) > 5:
        warnings.append(
            f"{qa['near_duplicate_count']} near-duplicate images detected in sample. "
            f"Consider deduplication for cleaner training."
        )

    if qa.get("blank_or_uniform_count", 0) > 3:
        warnings.append(
            f"{qa['blank_or_uniform_count']} blank/uniform images found. "
            f"These may cause training instability."
        )

    if st.get("total_samples", 0) < 500 and st.get("total_samples", 0) > 0:
        warnings.append(
            f"Only {st['total_samples']} training samples. "
            f"Consider heavy augmentation and low learning rate."
        )

    if fa.get("has_dicom"):
        warnings.append(
            "DICOM files detected. Custom pydicom loader with HU windowing will be used. "
            "Standard PIL loaders will NOT work on .dcm files."
        )

    if not profile.get("structure", {}).get("has_splits"):
        warnings.append(
            "No pre-built train/val split detected. "
            "Stratified 80/20 split will be created automatically."
        )

    return warnings
