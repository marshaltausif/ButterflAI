# modal/butterflai_worker.py  (v2 — full pipeline with Drive upload + Streamlit gen)
# Deploy once: modal deploy modal/butterflai_worker.py

import modal, os, json, subprocess, sys, time
from pathlib import Path

app = modal.App("butterflai")
vol = modal.Volume.from_name("butterflai-jobs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0", "torchvision==0.17.0", "timm==0.9.16",
        "datasets==2.19.0", "kaggle==1.6.14",
        "Pillow>=10.0.0", "tqdm>=4.66.0", "numpy>=1.26.0",
        "streamlit>=1.35.0", "matplotlib>=3.8.0",
    )
    .run_commands("apt-get install -y curl unzip")
)

secrets = [modal.Secret.from_name("butterflai-secrets", required=False)]


@app.function(
    image=image, gpu="T4", timeout=7200, memory=16384,
    volumes={"/jobs": vol}, secrets=secrets,
)
def butterflai_train(
    job_id: str, config: dict, train_py: str, classes: list,
    dataset_id: str, dataset_source: str,
    kaggle_user: str = "", kaggle_key: str = "",
    drive_folder_id: str = "", drive_access_token: str = "",
):
    import torch
    print(f"[ButterflAI] ══════════════════════════════════")
    print(f"[ButterflAI] Job: {job_id}")
    print(f"[ButterflAI] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU!)'}")
    print(f"[ButterflAI] Model: {config['model_name']} | Epochs: {config['epochs']} | Classes: {config['num_classes']}")
    print(f"[ButterflAI] Dataset: {dataset_source}/{dataset_id}")
    print(f"[ButterflAI] ══════════════════════════════════")
    t0 = time.time()

    job_dir = Path(f"/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    ds_dir  = job_dir / "dataset"
    ds_dir.mkdir(exist_ok=True)

    # Write all input files
    (job_dir / "config.json").write_text(json.dumps(config, indent=2))
    (job_dir / "classes.txt").write_text("\n".join(classes))
    (job_dir / "train.py").write_text(train_py)
    (job_dir / "job.json").write_text(json.dumps({
        "job_id": job_id, "status": "RUNNING",
        "model": config["model_name"], "dataset": dataset_id,
        "started_at": _now(), "gpu": "T4 (Modal)",
    }, indent=2))

    # Download dataset
    print(f"[ButterflAI] Downloading dataset…")
    if dataset_source == "kaggle":
        _setup_kaggle(kaggle_user or os.environ.get("KAGGLE_USERNAME",""),
                      kaggle_key  or os.environ.get("KAGGLE_KEY",""))
        _dl_kaggle(dataset_id, ds_dir)
    elif dataset_source == "hf":
        _dl_hf(dataset_id, ds_dir, classes)

    # Build manifest
    manifest = _build_manifest(ds_dir, classes, job_id, dataset_id)
    (job_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ButterflAI] Dataset ready — {manifest['total_images']} images across {len(classes)} classes")

    # Run training
    print("[ButterflAI] ── Training start ──────────────────")
    proc = subprocess.run(
        [sys.executable, str(job_dir / "train.py")],
        cwd=str(job_dir), timeout=6800,
    )
    ok = proc.returncode == 0

    # Parse best accuracy from history
    best_acc = 0.0
    hist_path = job_dir / "history.json"
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
            best_acc = max((h.get("val_acc", 0) for h in hist), default=0.0)
        except Exception: pass

    elapsed = int(time.time() - t0)
    print(f"[ButterflAI] ── Training {'DONE' if ok else 'FAILED'} in {elapsed}s ──")
    print(f"[ButterflAI] Best val_acc: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Generate Streamlit app
    st_code = _gen_streamlit(job_id, classes, config["model_name"],
                             config["image_size"], best_acc, dataset_id)
    (job_dir / "streamlit_app.py").write_text(st_code)
    print(f"[ButterflAI] streamlit_app.py written ({len(st_code):,} chars)")

    # Upload outputs to Google Drive
    if drive_access_token and drive_folder_id:
        print(f"[ButterflAI] Uploading outputs to Google Drive…")
        _upload_to_drive(job_dir, drive_folder_id, drive_access_token)
    else:
        print("[ButterflAI] No Drive credentials — outputs saved to Modal volume only")
        print(f"[ButterflAI] Retrieve with: modal volume get butterflai-jobs {job_id}/")

    # Finalize job.json
    (job_dir / "job.json").write_text(json.dumps({
        "job_id": job_id,
        "status": "DONE" if ok else "FAILED",
        "model": config["model_name"], "dataset": dataset_id,
        "best_val_acc": round(best_acc, 4),
        "elapsed_seconds": elapsed,
        "finished_at": _now(),
    }, indent=2))

    vol.commit()
    print(f"[ButterflAI] Job {job_id} complete. Volume committed.")
    return {"status": "DONE" if ok else "FAILED", "job_id": job_id, "best_acc": best_acc}


# ─── Kaggle ───────────────────────────────────────────────────────────────────
def _setup_kaggle(user, key):
    if user: os.environ["KAGGLE_USERNAME"] = user
    if key:  os.environ["KAGGLE_KEY"] = key
    d = Path.home() / ".kaggle"; d.mkdir(exist_ok=True)
    if user and key:
        p = d / "kaggle.json"
        p.write_text(json.dumps({"username": user, "key": key}))
        p.chmod(0o600)

def _dl_kaggle(dataset_id: str, dest: Path):
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"[ButterflAI] Kaggle note: {r.stderr[-200:]}")


# ─── HuggingFace ─────────────────────────────────────────────────────────────
def _dl_hf(dataset_id: str, dest: Path, classes: list):
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        ds = load_dataset(dataset_id)
        key = "train" if "train" in ds else list(ds.keys())[0]
        for item in ds[key]:
            lbl = item.get("label", 0)
            cls = classes[lbl] if isinstance(lbl, int) and lbl < len(classes) else str(lbl)
            d = dest / cls; d.mkdir(exist_ok=True)
            img = item.get("image") or item.get("img")
            if img:
                img.save(d / f"{len(list(d.iterdir())):05d}.jpg")
    except Exception as e:
        print(f"[ButterflAI] HF error: {e}")


# ─── Manifest ────────────────────────────────────────────────────────────────
def _build_manifest(ds_dir: Path, classes: list, job_id: str, ds_id: str) -> dict:
    images, exts = [], {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for cls in classes:
        d = ds_dir / cls
        if not d.exists():
            for sub in ds_dir.iterdir():
                if sub.is_dir() and sub.name.lower() == cls.lower(): d = sub; break
        if not d.exists(): continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in exts)
        si = max(1, int(len(files) * 0.8))
        for i, f in enumerate(files):
            images.append({"path": f"{cls}/{f.name}", "label": cls,
                           "split": "train" if i < si else "val"})
    return {"job_id": job_id, "dataset": ds_id, "classes": classes,
            "total_images": len(images), "images": images}


# ─── Drive upload ─────────────────────────────────────────────────────────────
def _upload_to_drive(job_dir: Path, folder_id: str, token: str):
    import urllib.request
    files = ["best_model.pth", "history.json", "streamlit_app.py",
             "config.json", "classes.txt", "job.json", "dataset_manifest.json"]
    for fname in files:
        p = job_dir / fname
        if not p.exists(): continue
        data = p.read_bytes()
        mime = ("application/octet-stream" if fname.endswith(".pth") else
                "application/json" if fname.endswith(".json") else
                "text/x-python" if fname.endswith(".py") else "text/plain")
        meta = json.dumps({"name": fname, "parents": [folder_id]}).encode()
        bnd  = b"bnd_butterflai"
        body = (b"--" + bnd + b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n" +
                meta + b"\r\n--" + bnd + b"\r\nContent-Type: " + mime.encode() +
                b"\r\n\r\n" + data + b"\r\n--" + bnd + b"--")
        req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            data=body,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": f"multipart/related; boundary={bnd.decode()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                res = json.loads(r.read())
                print(f"[ButterflAI] ✓ Drive: {fname} → {res.get('id','?')}")
        except Exception as e:
            print(f"[ButterflAI] Drive warn {fname}: {e}")


# ─── Streamlit generator ──────────────────────────────────────────────────────
def _gen_streamlit(job_id, classes, model_name, image_size, best_acc, dataset_name):
    return f'''import streamlit as st, torch, timm, json, os, numpy as np
from PIL import Image
from torchvision import transforms
import io, urllib.request

st.set_page_config(page_title="ButterflAI · {job_id}", page_icon="🦋", layout="centered")
st.markdown("""<style>
body,.stApp{{background:#0b0b0e;color:#ede9e2;font-family:'DM Sans',sans-serif}}
.pred-bar-wrap{{background:#1b1b22;border-radius:99px;height:8px;overflow:hidden;margin:4px 0}}
.pred-bar-fill{{height:100%;background:linear-gradient(90deg,#c8a84b,#2dd4bf);border-radius:99px}}
.result-box{{background:linear-gradient(135deg,rgba(200,168,75,.08),rgba(45,212,191,.05));border:1px solid rgba(200,168,75,.25);border-radius:14px;padding:24px;text-align:center;margin:20px 0}}
</style>""", unsafe_allow_html=True)

CLASSES = {repr(classes)}
IMAGE_SIZE = {image_size}

@st.cache_resource
def load_model():
    m = timm.create_model("{model_name}".replace("-","_"), pretrained=False, num_classes=len(CLASSES))
    if os.path.exists("best_model.pth"):
        ckpt = torch.load("best_model.pth", map_location="cpu")
        m.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    return m.eval()

tfm = transforms.Compose([transforms.Resize(int(IMAGE_SIZE*1.14)),transforms.CenterCrop(IMAGE_SIZE),
      transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

@torch.no_grad()
def predict(img):
    t = tfm(img.convert("RGB")).unsqueeze(0)
    p = torch.softmax(load_model()(t)[0], 0).numpy()
    top = np.argsort(p)[::-1][:5]
    return [(CLASSES[i], float(p[i])) for i in top]

st.markdown("# 🦋 ButterflAI Demo")
st.caption(f"**{model_name}** · {len(classes)} classes · **{best_acc*100:.1f}%** val accuracy · Trained on {dataset_name}")
st.divider()

col1, col2 = st.columns([1.3,1])
with col1:
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"])
    url = st.text_input("Or image URL", placeholder="https://…")
    run = st.button("🔍 Classify", use_container_width=True, type="primary")
with col2:
    for label, val, color in [("Architecture","{model_name}","#c8a84b"),("Val Accuracy",f"{best_acc*100:.1f}%","#2dd4bf"),("Classes",str(len(classes)),"#ede9e2")]:
        st.markdown(f\'\'\'<div style="background:#131318;border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:14px;margin-bottom:8px;text-align:center">
        <div style="font-size:10px;color:#5c5855;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">{{label}}</div>
        <div style="font-size:16px;font-weight:700;color:{{color}}">{{val}}</div></div>\'\'\', unsafe_allow_html=True)

if run and (up or url):
    try:
        img = Image.open(up) if up else Image.open(io.BytesIO(urllib.request.urlopen(url).read()))
        st.image(img, use_column_width=True)
        with st.spinner("Running…"):
            results = predict(img)
        top_cls, top_prob = results[0]
        st.markdown(f\'\'\'<div class="result-box">
          <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#9b9490;margin-bottom:8px">Prediction</div>
          <div style="font-size:28px;font-weight:700;margin-bottom:4px">{{top_cls}}</div>
          <div style="font-size:20px;font-weight:700;color:#2dd4bf">{{top_prob*100:.1f}}% confidence</div>
        </div>\'\'\', unsafe_allow_html=True)
        for cls, prob in results:
            pct = prob*100
            st.markdown(f\'\'\'<div style="margin:6px 0"><div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
            <span style="color:{{"#c8a84b" if cls==top_cls else "#ede9e2"}};font-weight:{{"600" if cls==top_cls else "400"}}">{{cls}}</span>
            <span style="color:#9b9490">{{pct:.2f}}%</span></div>
            <div class="pred-bar-wrap"><div class="pred-bar-fill" style="width:{{pct}}%"></div></div></div>\'\'\', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {{e}}")

with st.sidebar:
    st.markdown("### 🦋 ButterflAI"); st.caption(f"Job: `{job_id}`"); st.divider()
    st.markdown("**Classes:**")
    for i,c in enumerate(CLASSES): st.markdown(f"`{{i:02d}}` {{c}}")
    st.caption("butterflai.app")
'''


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
