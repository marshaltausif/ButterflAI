# modal/butterflai_worker.py
# Deploy once: modal deploy modal/butterflai_worker.py
#
# Real pipeline:
#  1. Download dataset (Kaggle datasets API + HuggingFace datasets library)
#  2. Data Intelligence Layer — physically reads every file, measures everything
#  3. Code Generator — writes train.py specific to this dataset's profile
#  4. Train — streams stdout back line by line
#  5. Upload outputs to Google Drive
#
# Supports all modalities: image, tabular, text, audio

import modal, os, json, subprocess, sys, time, shutil, io
from pathlib import Path

app = modal.App("butterflai")

# Persistent volume — job files survive across restarts
vol = modal.Volume.from_name("butterflai-jobs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "curl", "ffmpeg", "libsndfile1")
    .pip_install(
        # ── Core training ──────────────────────────────────────────────────
        "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
        "timm==0.9.16",
        # ── Data ──────────────────────────────────────────────────────────
        "datasets==2.19.0", "kaggle==1.6.14",
        "pandas==2.2.0", "pyarrow==15.0.0",
        "scikit-learn>=1.4.0",
        "pillow>=10.0.0",
        # ── Formats ───────────────────────────────────────────────────────
        "pydicom>=2.4.0",
        "tifffile>=2024.0.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        # ── Text / NLP ────────────────────────────────────────────────────
        "transformers==4.40.0",
        "tokenizers>=0.19.0",
        "evaluate>=0.4.0",
        "accelerate>=0.29.0",
        "sentencepiece>=0.2.0",
        # ── Utils ─────────────────────────────────────────────────────────
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "streamlit>=1.35.0",
    )
    .add_local_python_source("dataset_intelligence")
    .add_local_python_source("code_generator")
    .add_local_python_source("generate_streamlit")
)

secrets = [modal.Secret.from_name("butterflai-secrets", required=False)]


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    memory=16384,
    volumes={"/jobs": vol},
    secrets=secrets,
)
def butterflai_train(
    job_id: str,
    config: dict,
    declared_classes: list,
    dataset_id: str,
    dataset_source: str,          # "kaggle" | "hf"
    goal: str,
    ai_provider: str = "gemini",
    kaggle_user: str = "",
    kaggle_key: str  = "",
    hf_token: str    = "",
    drive_folder_id: str = "",
    drive_access_token: str = "",
) -> dict:
    """
    Full ButterflAI pipeline on Modal T4 GPU.
    Returns result dict; stdout is streamed in real time.
    """
    import torch
    from dataset_intelligence import build_dataset_profile
    from code_generator import generate_train_py
    from generate_streamlit import generate_streamlit_app

    t0 = time.time()

    print(f"[ButterflAI] ════════════════════════════════════════════════════")
    print(f"[ButterflAI] Job ID:    {job_id}")
    print(f"[ButterflAI] GPU:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[ButterflAI] CUDA:      {torch.version.cuda}")
    print(f"[ButterflAI] PyTorch:   {torch.__version__}")
    print(f"[ButterflAI] Goal:      {goal}")
    print(f"[ButterflAI] Dataset:   {dataset_source}/{dataset_id}")
    print(f"[ButterflAI] Model:     {config.get('model_name', '?')}")
    print(f"[ButterflAI] AI:        {ai_provider}")
    print(f"[ButterflAI] ════════════════════════════════════════════════════")

    job_dir = Path(f"/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    ds_dir  = job_dir / "dataset"
    ds_dir.mkdir(exist_ok=True)

    # ── Step 1: Download dataset ──────────────────────────────────────────────
    print(f"\n[ButterflAI] ── Step 1: Downloading {dataset_source}/{dataset_id}")
    download_ok, download_msg = False, ""

    if dataset_source == "kaggle":
        download_ok, download_msg = download_kaggle(
            dataset_id, ds_dir,
            kaggle_user or os.environ.get("KAGGLE_USERNAME", ""),
            kaggle_key  or os.environ.get("KAGGLE_KEY",      ""),
        )
    elif dataset_source == "hf":
        download_ok, download_msg = download_hf(
            dataset_id, ds_dir, declared_classes,
            hf_token or os.environ.get("HF_TOKEN", ""),
        )
    else:
        download_msg = f"Unknown source: {dataset_source}"

    total_files = sum(1 for _ in ds_dir.rglob("*") if _.is_file())
    print(f"[ButterflAI] Download: {'OK' if download_ok else 'WARN'} — {download_msg}")
    print(f"[ButterflAI] Files on disk: {total_files:,}")

    if total_files == 0:
        print("[ButterflAI] ERROR: No files downloaded. Aborting.")
        return {"status": "FAILED", "error": download_msg, "job_id": job_id}

    # ── Step 2: Data Intelligence Layer ──────────────────────────────────────
    print(f"\n[ButterflAI] ── Step 2: Data Intelligence Analysis")
    profile = build_dataset_profile(
        dataset_dir=str(ds_dir),
        goal=goal,
        declared_classes=declared_classes,
    )

    profile_path = job_dir / "dataset_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, default=str))
    print(f"[ButterflAI] Profile saved: {profile_path}")

    # Surface warnings immediately
    for w in profile.get("warnings", []):
        print(f"[DataIntel] ⚠ {w}")

    # ── Step 3: Patch config with real profile recommendations ────────────────
    print(f"\n[ButterflAI] ── Step 3: Patching config with DatasetProfile")
    recs = profile.get("recommendations", {})
    config = {
        **config,
        "dataset_root":  str(ds_dir),
        "modality":      profile.get("modality", "image"),
    }
    if recs.get("recommended_image_size"):
        old = config.get("image_size", 224)
        config["image_size"] = recs["recommended_image_size"]
        if old != config["image_size"]:
            print(f"[ButterflAI] image_size: {old} → {config['image_size']} (from profile)")

    if recs.get("target_column"):
        config["target_column"] = recs["target_column"]
        print(f"[ButterflAI] target_column: {recs['target_column']}")

    if recs.get("text_column"):
        config["text_column"]  = recs["text_column"]
        config["label_column"] = recs.get("label_column", "label")
        print(f"[ButterflAI] text_column: {recs['text_column']} | label_column: {recs['label_column']}")

    if recs.get("sample_rate"):
        config["sample_rate"] = recs["sample_rate"]
        print(f"[ButterflAI] sample_rate: {recs['sample_rate']}")

    config["classes"]     = declared_classes
    config["num_classes"] = len(declared_classes)

    (job_dir / "config.json").write_text(json.dumps(config, indent=2))
    (job_dir / "classes.txt").write_text("\n".join(declared_classes))

    # ── Step 4: Generate train.py from DatasetProfile ─────────────────────────
    print(f"\n[ButterflAI] ── Step 4: Generating train.py from DatasetProfile")
    train_py = generate_train_py(
        profile=profile, config=config, job_id=job_id, ai_provider=ai_provider
    )
    (job_dir / "train.py").write_text(train_py)
    print(f"[ButterflAI] train.py: {len(train_py):,} chars | "
          f"loader: {recs.get('loader_type','?')} | "
          f"aug: {recs.get('augmentation_strategy','?')} | "
          f"sampler: {recs.get('recommended_sampling','none')}")

    # ── Step 5: Write job metadata ─────────────────────────────────────────────
    (job_dir / "job.json").write_text(json.dumps({
        "job_id":        job_id,
        "status":        "RUNNING",
        "model":         config.get("model_name"),
        "dataset_id":    dataset_id,
        "dataset_source":dataset_source,
        "modality":      profile.get("modality"),
        "domain":        profile.get("domain_analysis", {}).get("type", "general"),
        "total_files":   total_files,
        "ai_provider":   ai_provider,
        "started_at":    _now(),
        "gpu":           "T4 (Modal.com)",
        "profile_warnings": profile.get("warnings", []),
        "config":        config,
    }, indent=2, default=str))

    # ── Step 6: Run training ───────────────────────────────────────────────────
    print(f"\n[ButterflAI] ── Step 6: Training")
    print(f"[ButterflAI] Running: python3 train.py")
    print(f"[ButterflAI] ─────────────────────────────────────────────────────")

    result = subprocess.run(
        [sys.executable, str(job_dir / "train.py")],
        cwd=str(job_dir),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(job_dir)},
        timeout=6800,
    )
    train_ok = result.returncode == 0
    elapsed  = int(time.time() - t0)

    print(f"\n[ButterflAI] ─────────────────────────────────────────────────────")
    print(f"[ButterflAI] Training {'DONE ✓' if train_ok else 'FAILED ✗'} in {elapsed}s")

    # Parse best accuracy
    best_acc = 0.0
    hist_path = job_dir / "history.json"
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
            if isinstance(hist, list):
                best_acc = max((h.get("val_acc", 0) for h in hist), default=0.0)
            elif isinstance(hist, dict):
                best_acc = hist.get("eval_accuracy", 0.0)
        except Exception:
            pass

    print(f"[ButterflAI] Best val_acc: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # ── Step 7: Generate Streamlit app ────────────────────────────────────────
    print(f"\n[ButterflAI] ── Step 7: Generating Streamlit demo")
    try:
        st_code = generate_streamlit_app(
            job_id=job_id, classes=declared_classes,
            model_name=config.get("model_name", "efficientnet_b3"),
            image_size=config.get("image_size", 224),
            best_acc=best_acc, dataset_name=dataset_id,
            modality=profile.get("modality", "image"),
        )
        (job_dir / "streamlit_app.py").write_text(st_code)
        print(f"[ButterflAI] streamlit_app.py written ({len(st_code):,} chars)")
    except Exception as e:
        print(f"[ButterflAI] Streamlit generation warning: {e}")

    # ── Step 8: Upload to Google Drive ────────────────────────────────────────
    drive_link = None
    if drive_access_token and drive_folder_id:
        print(f"\n[ButterflAI] ── Step 8: Uploading to Google Drive")
        drive_link = upload_to_drive(job_dir, drive_folder_id, drive_access_token)
    else:
        print(f"\n[ButterflAI] ── Step 8: No Drive credentials — outputs in Modal volume")
        print(f"[ButterflAI] Retrieve: modal volume get butterflai-jobs {job_id}/")

    # ── Finalize ───────────────────────────────────────────────────────────────
    final_status = "DONE" if train_ok else "FAILED"
    (job_dir / "job.json").write_text(json.dumps({
        "job_id":        job_id,
        "status":        final_status,
        "model":         config.get("model_name"),
        "dataset_id":    dataset_id,
        "modality":      profile.get("modality"),
        "best_val_acc":  round(best_acc, 4),
        "elapsed_seconds": elapsed,
        "ai_provider":   ai_provider,
        "drive_link":    drive_link,
        "finished_at":   _now(),
    }, indent=2))

    vol.commit()
    print(f"[ButterflAI] Job {job_id} committed to volume.")

    return {
        "status":    final_status,
        "job_id":    job_id,
        "best_acc":  best_acc,
        "elapsed":   elapsed,
        "modality":  profile.get("modality"),
        "warnings":  profile.get("warnings", []),
        "drive_link": drive_link,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOADERS
# ══════════════════════════════════════════════════════════════════════════════

def download_kaggle(dataset_id: str, dest: Path, user: str, key: str) -> tuple:
    if not user or not key:
        return False, "No Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY)"

    # Write credentials
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"]      = key
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    cred_file = kdir / "kaggle.json"
    cred_file.write_text(json.dumps({"username": user, "key": key}))
    cred_file.chmod(0o600)

    # Try as dataset (most common)
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode == 0:
        return True, f"Kaggle dataset '{dataset_id}' downloaded"

    # Try as competition
    comp_name = dataset_id.split("/")[-1]
    r2 = subprocess.run(
        ["kaggle", "competitions", "download", "-c", comp_name, "-p", str(dest)],
        capture_output=True, text=True, timeout=600,
    )
    if r2.returncode == 0:
        # Unzip competition files
        for zf in dest.glob("*.zip"):
            subprocess.run(["unzip", "-q", str(zf), "-d", str(dest)])
            zf.unlink()
        return True, f"Kaggle competition '{comp_name}' downloaded"

    stderr = (r.stderr + r2.stderr).strip()[-500:]
    return False, f"Kaggle download failed: {stderr}"


def download_hf(dataset_id: str, dest: Path,
                declared_classes: list, hf_token: str = "") -> tuple:
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage

        kwargs = {"trust_remote_code": True}
        if hf_token:
            kwargs["token"] = hf_token

        print(f"[ButterflAI] HuggingFace: loading {dataset_id}…")
        ds = load_dataset(dataset_id, **kwargs)

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split = ds[split_name]
        features = split.features
        print(f"[ButterflAI] HF: {len(split):,} samples | features: {list(features.keys())}")

        # Detect column types
        img_col   = None
        label_col = None
        text_col  = None
        audio_col = None

        for col, feat in features.items():
            ft = type(feat).__name__
            cl = col.lower()
            if "Image" in ft:
                img_col = col
            elif "Audio" in ft or "audio" in cl:
                audio_col = col
            elif "ClassLabel" in ft or cl in ("label","labels","target","class","category"):
                label_col = col
            elif cl in ("text","sentence","input","content","review","comment","passage"):
                text_col = col

        print(f"[ButterflAI] HF: img={img_col} audio={audio_col} text={text_col} label={label_col}")

        # Get class names
        if label_col and hasattr(features[label_col], "names"):
            cls_names = features[label_col].names
        elif declared_classes:
            cls_names = declared_classes
        else:
            # Infer from unique values
            cls_names = [str(v) for v in sorted(set(split[label_col][:5000]))] if label_col else ["class_0"]

        print(f"[ButterflAI] HF: {len(cls_names)} classes: {cls_names[:5]}{'…' if len(cls_names)>5 else ''}")
        saved = 0

        # ── Image classification ──────────────────────────────────────────────
        if img_col is not None:
            for item in split:
                label_raw = item.get(label_col, 0) if label_col else 0
                if isinstance(label_raw, int) and label_raw < len(cls_names):
                    cls = cls_names[label_raw]
                else:
                    cls = str(label_raw).replace("/", "_").replace(" ", "_")

                cls_dir = dest / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                idx = len(list(cls_dir.iterdir()))

                img = item[img_col]
                try:
                    if isinstance(img, PILImage.Image):
                        img.save(cls_dir / f"{idx:06d}.jpg")
                        saved += 1
                    elif isinstance(img, bytes):
                        PILImage.open(io.BytesIO(img)).convert("RGB").save(cls_dir / f"{idx:06d}.jpg")
                        saved += 1
                    elif isinstance(img, dict) and "bytes" in img:
                        PILImage.open(io.BytesIO(img["bytes"])).convert("RGB").save(cls_dir / f"{idx:06d}.jpg")
                        saved += 1
                except Exception:
                    pass

        # ── Text classification ───────────────────────────────────────────────
        elif text_col is not None:
            jsonl_path = dest / "train.jsonl"
            with open(jsonl_path, "w") as f:
                for item in split:
                    row = {
                        text_col:  str(item.get(text_col, "")),
                        "label":   str(item.get(label_col, 0)) if label_col else "0",
                    }
                    f.write(json.dumps(row) + "\n")
                    saved += 1

        # ── Audio classification ──────────────────────────────────────────────
        elif audio_col is not None:
            import soundfile as sf, numpy as np
            for item in split:
                label_raw = item.get(label_col, 0) if label_col else 0
                cls = cls_names[label_raw] if isinstance(label_raw, int) and label_raw < len(cls_names) else str(label_raw)
                cls_dir = dest / cls.replace("/","_").replace(" ","_")
                cls_dir.mkdir(parents=True, exist_ok=True)
                idx = len(list(cls_dir.iterdir()))
                audio = item[audio_col]
                try:
                    arr = audio.get("array", np.zeros(16000))
                    sr  = audio.get("sampling_rate", 16000)
                    sf.write(cls_dir / f"{idx:06d}.wav", arr, sr)
                    saved += 1
                except Exception:
                    pass

        # ── Generic (tabular) ─────────────────────────────────────────────────
        else:
            import pandas as pd
            pd.DataFrame(split).to_parquet(dest / "data.parquet")
            saved = len(split)

        print(f"[ButterflAI] HF: {saved:,} samples saved to disk")
        return (saved > 0), f"HuggingFace '{dataset_id}': {saved:,} samples"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"HuggingFace error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_drive(job_dir: Path, folder_id: str, token: str) -> str | None:
    import urllib.request

    files_to_upload = [
        "best_model.pth",   "history.json",     "streamlit_app.py",
        "config.json",      "classes.txt",      "job.json",
        "dataset_profile.json", "train.py",
    ]

    folder_link = None
    uploaded    = 0

    for fname in files_to_upload:
        fpath = job_dir / fname
        if not fpath.exists():
            continue

        data = fpath.read_bytes()
        mime = {
            ".pth":  "application/octet-stream",
            ".json": "application/json",
            ".py":   "text/x-python",
            ".txt":  "text/plain",
        }.get(fpath.suffix, "application/octet-stream")

        meta = json.dumps({"name": fname, "parents": [folder_id]}).encode()
        bnd  = b"bnd_butterflai"
        body = (
            b"--" + bnd + b"\r\nContent-Type: application/json\r\n\r\n" +
            meta + b"\r\n--" + bnd + b"\r\nContent-Type: " + mime.encode() +
            b"\r\n\r\n" + data + b"\r\n--" + bnd + b"--"
        )

        req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  f"multipart/related; boundary={bnd.decode()}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                fid = result.get("id", "?")
                print(f"[ButterflAI] Drive ✓ {fname} → {fid}")
                if not folder_link:
                    folder_link = f"https://drive.google.com/drive/folders/{folder_id}"
                uploaded += 1
        except Exception as e:
            print(f"[ButterflAI] Drive ✗ {fname}: {e}")

    print(f"[ButterflAI] Drive: {uploaded} files uploaded → {folder_link}")
    return folder_link


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    print("Deploy: modal deploy modal/butterflai_worker.py")
    print("Secrets: modal secret create butterflai-secrets KAGGLE_USERNAME=x KAGGLE_KEY=x")
