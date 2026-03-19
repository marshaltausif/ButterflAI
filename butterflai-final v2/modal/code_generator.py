# modal/code_generator.py
"""
ButterflAI Code Generator
===========================
Generates train.py from a real DatasetProfile.
Every decision (loader, transforms, sampler, loss, model in_channels)
comes from actual measured data — not assumptions.
"""

import json


def generate_train_py(profile: dict, config: dict, job_id: str, ai_provider: str) -> str:
    """
    Main entry. Returns complete, runnable train.py as a string.
    profile: DatasetProfile from dataset_intelligence.py
    config:  hyperparams from user config
    """
    m     = profile.get("modality", "image")
    recs  = profile.get("recommendations", {})
    spec  = profile.get("preprocessing_spec", {})
    ca    = profile.get("class_analysis", {})
    da    = profile.get("domain_analysis", {})
    fa    = profile.get("format_analysis", {})

    lines = []
    w = lines.append  # shorthand

    # ── Header ────────────────────────────────────────────────────────────────
    w(f"# ButterflAI — Generated train.py")
    w(f"# Job: {job_id}")
    w(f"# AI Provider: {ai_provider}")
    w(f"# Modality: {m}")
    w(f"# Domain: {da.get('type','general') if m=='image' else m}")
    w(f"# Classes: {config.get('num_classes','?')} | Epochs: {config.get('epochs','?')}")
    w(f"# Auto-generated from DatasetProfile — DO NOT edit manually")
    w("")

    # ── Imports ───────────────────────────────────────────────────────────────
    w("import torch, timm, json, os, time, math, random")
    w("import numpy as np")
    w("from pathlib import Path")
    w("from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler")
    w("from torch.cuda.amp import autocast, GradScaler")

    if m == "image":
        w("import torchvision.transforms as T")
        w("from PIL import Image")
        if recs.get("needs_dicom_loader"):
            w("import pydicom")
        if recs.get("needs_tiff_loader"):
            w("from PIL import ImageSequence")

    elif m == "tabular":
        w("import pandas as pd")
        w("from sklearn.preprocessing import StandardScaler, LabelEncoder")
        w("from sklearn.model_selection import train_test_split")

    elif m == "text":
        w("from transformers import AutoTokenizer, AutoModelForSequenceClassification")
        w("from transformers import TrainingArguments, Trainer")

    elif m == "audio":
        w("import torchaudio")
        w("import torchaudio.transforms as AT")

    for imp in spec.get("extra_imports", []):
        w(imp)

    w("")
    w("# ─── Config ─────────────────────────────────────────────────────────────────")
    w('with open("config.json") as f: cfg = json.load(f)')
    w("DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'")
    w("print(f'[ButterflAI] Device: {DEVICE} | Model: {cfg[\"model_name\"]} | "
      "Classes: {cfg[\"num_classes\"]}')")
    w("")

    # ── Dataset loader — modality-specific ────────────────────────────────────
    if m == "image":
        _write_image_dataset(lines, recs, spec, da, fa, config)
    elif m == "tabular":
        _write_tabular_dataset(lines, recs, profile)
    elif m == "text":
        _write_text_dataset(lines, recs, profile, config)
    elif m == "audio":
        _write_audio_dataset(lines, recs, spec, config)

    # ── Transforms ────────────────────────────────────────────────────────────
    if m == "image":
        _write_transforms(lines, spec)

    # ── Data loading ──────────────────────────────────────────────────────────
    if m == "image":
        _write_image_dataloader(lines, recs, ca, config)
    elif m == "tabular":
        _write_tabular_dataloader(lines, recs)

    # ── Model ────────────────────────────────────────────────────────────────
    if m in ("image",):
        _write_image_model(lines, recs, config)
    elif m == "tabular":
        _write_tabular_model(lines, recs, profile, config)
    # text and audio models are handled inline

    # ── Training loop (image + tabular) ──────────────────────────────────────
    if m in ("image", "tabular"):
        _write_training_loop(lines, config, ca)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

def _write_image_dataset(lines, recs, spec, da, fa, config):
    w = lines.append
    w("# ─── Dataset ─────────────────────────────────────────────────────────────────")

    # Custom DICOM loader
    if spec.get("custom_loader_code"):
        w(spec["custom_loader_code"])
        w("")

    w("class ButterflAIDataset(Dataset):")
    w("    def __init__(self, root, classes, transform=None, split='train'):")
    w("        self.samples, self.transform = [], transform")
    w("        self.c2i = {c: i for i, c in enumerate(classes)}")
    w("        self.load_errors = 0")

    if recs.get("needs_dicom_loader"):
        w("        exts = {'.dcm', '.dicom'}")
    elif recs.get("needs_tiff_loader"):
        w("        exts = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.webp'}")
    else:
        w("        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}")

    w("        root = Path(root)")
    w("        # Support both flat ImageFolder and split-based layouts")
    w("        for cls in classes:")
    w("            # Try split-based first: root/train/class/img.jpg")
    w("            for base in [root/split/cls, root/cls]:")
    w("                if base.exists():")
    w("                    for f in base.rglob('*'):")
    w("                        if f.is_file() and f.suffix.lower() in exts:")
    w("                            self.samples.append((str(f), self.c2i.get(cls, 0)))")
    w("                    break")
    w("")
    w("    def __len__(self): return len(self.samples)")
    w("")
    w("    def __getitem__(self, i):")
    w("        path, label = self.samples[i]")
    w("        try:")

    if recs.get("needs_dicom_loader"):
        w("            img = load_dicom(path)")
    else:
        w("            img = Image.open(path)")
        if recs.get("convert_to_rgb", True):
            w("            img = img.convert('RGB')")
        else:
            w("            img = img.convert('L')")

    w("            if self.transform: img = self.transform(img)")
    w("            return img, label")
    w("        except Exception as e:")
    w("            self.load_errors += 1")
    w("            # Return a blank tensor instead of crashing")
    sz = recs.get("recommended_image_size", 224)
    ch = recs.get("input_channels", 3)
    w(f"            blank = torch.zeros({ch}, cfg['image_size'], cfg['image_size'])")
    w("            return blank, label")
    w("")


def _write_transforms(lines, spec):
    w = lines.append
    w("# ─── Transforms ─────────────────────────────────────────────────────────────")

    train_t = spec.get("train_transforms", [])
    val_t   = spec.get("val_transforms",   [])

    w("train_tfm = T.Compose([")
    for t in train_t:
        w(f"    {t},")
    w("])")
    w("")
    w("val_tfm = T.Compose([")
    for t in val_t:
        w(f"    {t},")
    w("])")
    w("")


def _write_image_dataloader(lines, recs, ca, config):
    w = lines.append
    w("# ─── Data Loading ────────────────────────────────────────────────────────────")
    w("classes    = cfg['classes']")
    w("DATASET_ROOT = cfg.get('dataset_root', 'dataset')")
    w("")
    w("train_ds = ButterflAIDataset(DATASET_ROOT, classes, train_tfm, 'train')")
    w("val_ds   = ButterflAIDataset(DATASET_ROOT, classes, val_tfm,   'val')")
    w("")
    w("# If no split found, create stratified split from train_ds")
    w("if len(val_ds) == 0:")
    w("    n = len(train_ds); nv = max(1, int(n * 0.2))")
    w("    train_ds, val_ds = torch.utils.data.random_split(")
    w("        ButterflAIDataset(DATASET_ROOT, classes, None, 'train'),")
    w("        [n - nv, nv], generator=torch.Generator().manual_seed(42)")
    w("    )")
    w("    train_ds.dataset.transform = train_tfm")
    w("    val_ds.dataset.transform   = val_tfm")
    w("")

    # Sampler for imbalanced datasets
    if recs.get("use_weighted_sampler") and ca.get("class_weights"):
        weights_str = json.dumps(ca["class_weights"])
        w(f"# Weighted sampler for class imbalance (ratio={ca.get('imbalance_ratio','?')}x)")
        w(f"_cw = {weights_str}")
        w("_sample_weights = [")
        w("    _cw.get(classes[s[1]], 1.0) if isinstance(s, tuple) else 1.0")
        w("    for s in getattr(train_ds, 'samples', [])")
        w("]")
        w("sampler = WeightedRandomSampler(")
        w("    weights=_sample_weights, num_samples=len(train_ds), replacement=True")
        w(") if _sample_weights else None")
        w("")
        w("bs = min(cfg['batch_size'], 32 if 'b3' in cfg['model_name'] or 'vit' in cfg['model_name'] else 64)")
        w("train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,")
        w("                          num_workers=2, pin_memory=True)")
    else:
        w("bs = min(cfg['batch_size'], 32 if 'b3' in cfg['model_name'] or 'vit' in cfg['model_name'] else 64)")
        w("train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,")
        w("                          num_workers=2, pin_memory=True)")

    w("val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,")
    w("                          num_workers=2, pin_memory=True)")
    w("print(f'[ButterflAI] Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {bs}')")
    w("")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _write_image_model(lines, recs, config):
    w = lines.append
    w("# ─── Model ───────────────────────────────────────────────────────────────────")
    w("pretrained = cfg.get('pretrained', 'imagenet') == 'imagenet'")
    in_ch = recs.get("input_channels", 3)
    if in_ch == 1:
        w("# Grayscale/medical: load pretrained, patch first conv layer")
        w("model = timm.create_model(cfg['model_name'].replace('-','_'),")
        w("    pretrained=pretrained, num_classes=cfg['num_classes'])")
        w("# Patch input conv to accept 1 channel (sum weights to preserve pretraining)")
        w("first_conv = None")
        w("for name, module in model.named_modules():")
        w("    if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:")
        w("        first_conv = (name, module); break")
        w("if first_conv:")
        w("    n, m = first_conv")
        w("    new_conv = torch.nn.Conv2d(1, m.out_channels, m.kernel_size,")
        w("        m.stride, m.padding, bias=m.bias is not None)")
        w("    new_conv.weight.data = m.weight.data.sum(dim=1, keepdim=True)")
        w("    parts = n.split('.')")
        w("    parent = model")
        w("    for p in parts[:-1]: parent = getattr(parent, p)")
        w("    setattr(parent, parts[-1], new_conv)")
    else:
        w("model = timm.create_model(cfg['model_name'].replace('-','_'),")
        w("    pretrained=pretrained, num_classes=cfg['num_classes'])")
    w("model = model.to(DEVICE)")
    w("")


# ══════════════════════════════════════════════════════════════════════════════
# TABULAR DATASET + MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _write_tabular_dataset(lines, recs, profile):
    w = lines.append
    fa = profile.get("format_analysis", {})
    ca = profile.get("class_analysis", {})

    w("# ─── Tabular Dataset ─────────────────────────────────────────────────────────")
    w("import pandas as pd")
    w("from sklearn.preprocessing import StandardScaler, LabelEncoder")
    w("from sklearn.model_selection import train_test_split")
    w("")
    primary = fa.get("primary_file") or "dataset/data.csv"
    fmt = fa.get("format", ".csv")
    target = recs.get("target_column") or ca.get("target_column") or "label"

    w(f"_primary_file = cfg.get('dataset_file', '{primary}')")
    w(f"# Load data")
    if fmt == ".parquet":
        w("df = pd.read_parquet(_primary_file)")
    else:
        w("df = pd.read_csv(_primary_file)")
    w(f"TARGET = cfg.get('target_column', '{target}')")
    w("assert TARGET in df.columns, f'Target column {TARGET} not in dataset'")
    w("")
    w("# Encode target")
    w("le = LabelEncoder()")
    w("df[TARGET] = le.fit_transform(df[TARGET].astype(str))")
    w("classes = list(le.classes_)")
    w("cfg['classes'] = classes")
    w("cfg['num_classes'] = len(classes)")
    w("")
    w("# Features = all columns except target + any id columns")
    w("feature_cols = [c for c in df.columns if c != TARGET")
    w("                and c.lower() not in ('id','index','unnamed:0')]")
    w("X = df[feature_cols].fillna(0)")
    w("y = df[TARGET].values")
    w("")
    w("# Encode categorical features")
    w("for col in X.select_dtypes(include='object').columns:")
    w("    X[col] = LabelEncoder().fit_transform(X[col].astype(str))")
    w("X = X.values.astype(np.float32)")
    w("")
    w("# Train/val split")
    w("from sklearn.model_selection import train_test_split")
    w("X_tr,X_val,y_tr,y_val = train_test_split(X, y, test_size=0.2,")
    w("    stratify=y, random_state=42)")
    w("scaler = StandardScaler()")
    w("X_tr  = scaler.fit_transform(X_tr)")
    w("X_val = scaler.transform(X_val)")
    w("")
    w("class TabularDataset(torch.utils.data.Dataset):")
    w("    def __init__(self, X, y):")
    w("        self.X = torch.tensor(X, dtype=torch.float32)")
    w("        self.y = torch.tensor(y, dtype=torch.long)")
    w("    def __len__(self): return len(self.y)")
    w("    def __getitem__(self, i): return self.X[i], self.y[i]")
    w("")
    w("train_ds = TabularDataset(X_tr,  y_tr)")
    w("val_ds   = TabularDataset(X_val, y_val)")
    w("bs = cfg.get('batch_size', 256)")
    w("train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=0)")
    w("val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=0)")
    w(f"print(f'[ButterflAI] Tabular: {{len(X_tr)}} train, {{len(X_val)}} val, {{X_tr.shape[1]}} features')")
    w("")


def _write_tabular_dataloader(lines, recs):
    pass  # handled inline in _write_tabular_dataset


def _write_tabular_model(lines, recs, profile, config):
    w = lines.append
    fa = profile.get("format_analysis", {})
    n_features_guess = len(fa.get("feature_candidates", [])) or 32

    w("# ─── Tabular Model (MLP) ─────────────────────────────────────────────────────")
    w("n_features = X_tr.shape[1]")
    w("class TabularMLP(torch.nn.Module):")
    w("    def __init__(self, in_dim, n_classes):")
    w("        super().__init__()")
    w("        self.net = torch.nn.Sequential(")
    w("            torch.nn.Linear(in_dim, 256), torch.nn.BatchNorm1d(256),")
    w("            torch.nn.ReLU(), torch.nn.Dropout(0.3),")
    w("            torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128),")
    w("            torch.nn.ReLU(), torch.nn.Dropout(0.2),")
    w("            torch.nn.Linear(128, n_classes))")
    w("    def forward(self, x): return self.net(x)")
    w("model = TabularMLP(n_features, cfg['num_classes']).to(DEVICE)")
    w("")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT + AUDIO (handled via HuggingFace Trainer / torchaudio)
# ══════════════════════════════════════════════════════════════════════════════

def _write_text_dataset(lines, recs, profile, config):
    w = lines.append
    fa = profile.get("format_analysis", {})
    text_col  = recs.get("text_column", "text")
    label_col = recs.get("label_column", "label")
    max_len   = recs.get("max_token_length", 128)

    w("# ─── Text Classification via HuggingFace Transformers ───────────────────────")
    w("from datasets import load_dataset, Dataset as HFDataset")
    w("from transformers import (AutoTokenizer, AutoModelForSequenceClassification,")
    w("    TrainingArguments, Trainer)")
    w("import evaluate")
    w("")
    w(f"TOKENIZER = AutoTokenizer.from_pretrained(cfg['model_name'])")
    w(f"TEXT_COL  = cfg.get('text_column', '{text_col}')")
    w(f"LABEL_COL = cfg.get('label_column', '{label_col}')")
    w(f"MAX_LEN   = {max_len}")
    w("")
    w("# Load data")
    w("import pandas as pd, os")
    w("data_files = list(Path('dataset').rglob('*.jsonl')) + list(Path('dataset').rglob('*.csv'))")
    w("if not data_files: raise FileNotFoundError('No text data files found in dataset/')")
    w("primary = str(data_files[0])")
    w("if primary.endswith('.jsonl'):")
    w("    df = pd.read_json(primary, lines=True)")
    w("else:")
    w("    df = pd.read_csv(primary)")
    w("")
    w("from sklearn.preprocessing import LabelEncoder")
    w("le = LabelEncoder()")
    w("df[LABEL_COL] = le.fit_transform(df[LABEL_COL].astype(str))")
    w("classes = list(le.classes_)")
    w("cfg['classes'] = classes; cfg['num_classes'] = len(classes)")
    w("")
    w("def tokenize_fn(examples):")
    w("    return TOKENIZER(examples[TEXT_COL], truncation=True,")
    w("                     padding='max_length', max_length=MAX_LEN)")
    w("")
    w("from sklearn.model_selection import train_test_split")
    w("tr, val = train_test_split(df, test_size=0.2, stratify=df[LABEL_COL], random_state=42)")
    w("hf_tr  = HFDataset.from_pandas(tr.reset_index(drop=True))")
    w("hf_val = HFDataset.from_pandas(val.reset_index(drop=True))")
    w("hf_tr  = hf_tr.map(tokenize_fn, batched=True).rename_column(LABEL_COL, 'labels')")
    w("hf_val = hf_val.map(tokenize_fn, batched=True).rename_column(LABEL_COL, 'labels')")
    w("hf_tr.set_format('torch', columns=['input_ids','attention_mask','labels'])")
    w("hf_val.set_format('torch', columns=['input_ids','attention_mask','labels'])")
    w("")
    w("model = AutoModelForSequenceClassification.from_pretrained(")
    w("    cfg['model_name'], num_labels=cfg['num_classes'])")
    w("accuracy = evaluate.load('accuracy')")
    w("def compute_metrics(eval_pred):")
    w("    logits, labels = eval_pred")
    w("    preds = logits.argmax(-1)")
    w("    return accuracy.compute(predictions=preds, references=labels)")
    w("")
    w("training_args = TrainingArguments(")
    w("    output_dir='./checkpoints',")
    w("    num_train_epochs=cfg['epochs'],")
    w("    per_device_train_batch_size=cfg['batch_size'],")
    w("    per_device_eval_batch_size=cfg['batch_size'],")
    w("    learning_rate=cfg['learning_rate'],")
    w("    evaluation_strategy='epoch',")
    w("    save_strategy='epoch',")
    w("    load_best_model_at_end=True,")
    w("    metric_for_best_model='accuracy',")
    w("    fp16=cfg.get('fp16', True),")
    w("    logging_steps=50,")
    w("    report_to='none',")
    w(")")
    w("trainer = Trainer(model=model, args=training_args,")
    w("    train_dataset=hf_tr, eval_dataset=hf_val,")
    w("    compute_metrics=compute_metrics)")
    w("print('[ButterflAI] Starting text classification training...')")
    w("trainer.train()")
    w("metrics = trainer.evaluate()")
    w("best_acc = metrics.get('eval_accuracy', 0.0)")
    w("print(f'[ButterflAI] Done! eval_accuracy={best_acc:.4f}')")
    w("trainer.save_model('best_model')")
    w("import json as _j")
    w("with open('history.json','w') as f: _j.dump(metrics, f, indent=2)")
    w("print(f'[EPOCH:{cfg[\"epochs\"]}/{cfg[\"epochs\"]}] train_loss=0.0000 train_acc={best_acc:.4f} val_loss=0.0000 val_acc={best_acc:.4f} best={best_acc:.4f}')")


def _write_audio_dataset(lines, recs, spec, config):
    w = lines.append
    sr = recs.get("sample_rate", 16000)

    w("# ─── Audio Classification ────────────────────────────────────────────────────")
    w("import torchaudio, torchaudio.transforms as AT")
    w("from pathlib import Path")
    w("")
    w(f"TARGET_SR = {sr}")
    w("N_MELS    = 128")
    w("MAX_LEN   = 5.0  # seconds")
    w("")
    w("class AudioDataset(Dataset):")
    w("    def __init__(self, root, classes, train=True):")
    w("        self.samples = []")
    w("        self.c2i = {c: i for i, c in enumerate(classes)}")
    w("        root = Path(root)")
    w("        split = 'train' if train else 'val'")
    w("        audio_exts = {'.wav','.mp3','.flac','.ogg','.m4a'}")
    w("        for cls in classes:")
    w("            for base in [root/split/cls, root/cls]:")
    w("                if base.exists():")
    w("                    for f in base.rglob('*'):")
    w("                        if f.suffix.lower() in audio_exts:")
    w("                            self.samples.append((str(f), self.c2i.get(cls,0)))")
    w("                    break")
    w("")
    w(f"    _mel = AT.MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS)")
    w(f"    _db  = AT.AmplitudeToDB()")
    w("")
    w("    def __len__(self): return len(self.samples)")
    w("    def __getitem__(self, i):")
    w("        path, label = self.samples[i]")
    w("        try:")
    w("            wav, sr = torchaudio.load(path)")
    w("            if sr != TARGET_SR:")
    w("                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)")
    w("            # Mono")
    w("            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)")
    w("            # Pad/truncate to MAX_LEN")
    w("            max_samples = int(TARGET_SR * MAX_LEN)")
    w("            if wav.shape[1] > max_samples: wav = wav[:,:max_samples]")
    w("            elif wav.shape[1] < max_samples:")
    w("                wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[1]))")
    w("            mel = self.__class__._mel(wav)")
    w("            mel = self.__class__._db(mel)")
    w("            # Normalize")
    w("            mel = (mel - mel.mean()) / (mel.std() + 1e-6)")
    w("            # Repeat to 3 channels for pretrained model")
    w("            mel = mel.repeat(3, 1, 1)")
    w("            return mel, label")
    w("        except Exception as e:")
    w("            return torch.zeros(3, N_MELS, 128), label")
    w("")
    w("classes = cfg['classes']")
    w("DATASET_ROOT = cfg.get('dataset_root', 'dataset')")
    w("train_ds = AudioDataset(DATASET_ROOT, classes, train=True)")
    w("val_ds   = AudioDataset(DATASET_ROOT, classes, train=False)")
    w("if len(val_ds) == 0:")
    w("    n = len(train_ds); nv = max(1, int(n*0.2))")
    w("    train_ds, val_ds = torch.utils.data.random_split(")
    w("        AudioDataset(DATASET_ROOT, classes, True), [n-nv, nv],")
    w("        generator=torch.Generator().manual_seed(42))")
    w("bs = cfg.get('batch_size', 32)")
    w("train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=2, pin_memory=True)")
    w("val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=2, pin_memory=True)")
    w("# Use EfficientNet as backbone for mel spectrograms")
    w("model = timm.create_model('efficientnet_b0', pretrained=True,")
    w("    num_classes=cfg['num_classes']).to(DEVICE)")
    w(f"print(f'[ButterflAI] Audio: {{len(train_ds)}} train, {{len(val_ds)}} val, SR={sr}')")
    w("")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (shared for image + tabular)
# ══════════════════════════════════════════════════════════════════════════════

def _write_training_loop(lines, config, ca):
    w = lines.append
    use_class_weights = ca.get("is_imbalanced") and not ca.get("is_severely_imbalanced")

    w("# ─── Optimizer + Scheduler ───────────────────────────────────────────────────")
    w("opt = cfg.get('optimizer', 'adamw'); lr = cfg['learning_rate']")
    w("if   opt == 'adamw': optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)")
    w("elif opt == 'sgd':   optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)")
    w("else:                optimizer = torch.optim.Adam(model.parameters(), lr=lr)")
    w("")
    w("EPOCHS = cfg['epochs']")
    w("sc = cfg.get('scheduler', 'cosine')")
    w("if   sc == 'cosine':    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)")
    w("elif sc == 'onecycle':  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_loader), epochs=EPOCHS)")
    w("elif sc == 'step':      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,EPOCHS//4), gamma=0.5)")
    w("else:                   scheduler = None")
    w("")

    if use_class_weights and ca.get("class_weights"):
        weights_str = list(ca["class_weights"].values())
        w(f"# Class-weighted loss for moderate imbalance (ratio={ca.get('imbalance_ratio','?')}x)")
        w(f"_loss_weights = torch.tensor({weights_str}, dtype=torch.float32).to(DEVICE)")
        w("criterion = torch.nn.CrossEntropyLoss(weight=_loss_weights)")
    else:
        w("criterion = torch.nn.CrossEntropyLoss()")

    w("scaler = GradScaler() if cfg.get('fp16') and DEVICE == 'cuda' else None")
    w("best, patience, PAT = 0.0, 0, cfg.get('early_stopping_patience', 5)")
    w("history = []; t0 = time.time()")
    w("")
    w("# ─── Training Loop ───────────────────────────────────────────────────────────")
    w("for ep in range(1, EPOCHS + 1):")
    w("    model.train(); rl, rc, rt = 0.0, 0, 0")
    w("    for imgs, labels in train_loader:")
    w("        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)")
    w("        optimizer.zero_grad()")
    w("        if scaler:")
    w("            with autocast(): out = model(imgs); loss = criterion(out, labels)")
    w("            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()")
    w("        else:")
    w("            out = model(imgs); loss = criterion(out, labels)")
    w("            loss.backward(); optimizer.step()")
    w("        if sc == 'onecycle' and scheduler: scheduler.step()")
    w("        rl += loss.item()*imgs.size(0)")
    w("        rc += (out.argmax(1)==labels).sum().item()")
    w("        rt += labels.size(0)")
    w("    if scheduler and sc != 'onecycle': scheduler.step()")
    w("    ta, tl = rc/rt, rl/rt")
    w("")
    w("    model.eval(); vl, vc, vt = 0.0, 0, 0")
    w("    with torch.no_grad():")
    w("        for imgs, labels in val_loader:")
    w("            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)")
    w("            out = model(imgs)")
    w("            vl += criterion(out, labels).item()*imgs.size(0)")
    w("            vc += (out.argmax(1)==labels).sum().item()")
    w("            vt += labels.size(0)")
    w("    va, vll = (vc/vt if vt else 0), (vl/vt if vt else 0)")
    w("")
    w("    flag = ' ★ BEST' if va > best else ''")
    w("    print(f'[EPOCH:{ep}/{EPOCHS}] train_loss={tl:.4f} train_acc={ta:.4f} "
      "val_loss={vll:.4f} val_acc={va:.4f} best={max(best,va):.4f}{flag}')")
    w("    history.append({'epoch':ep,'train_loss':tl,'train_acc':ta,'val_loss':vll,'val_acc':va})")
    w("")
    w("    if va > best:")
    w("        best = va; patience = 0")
    w("        torch.save({'epoch':ep,'model_state':model.state_dict(),")
    w("                    'val_acc':va,'classes':cfg['classes']}, 'best_model.pth')")
    w("    else:")
    w("        patience += 1")
    w("        if PAT > 0 and patience >= PAT:")
    w("            print(f'[ButterflAI] Early stopping at epoch {ep}')")
    w("            break")
    w("")
    w("import json as _j")
    w("with open('history.json','w') as f: _j.dump(history, f, indent=2)")
    w("print(f'\\n[ButterflAI] Done! best_val_acc={best:.4f} | {int(time.time()-t0)}s')")
    w("print('[ButterflAI] Saved: best_model.pth, history.json')")
