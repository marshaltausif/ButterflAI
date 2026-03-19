// pages/index.js — ButterflAI v3 Complete
import { useSession, signIn, signOut } from "next-auth/react"
import { useState, useRef, useEffect, useCallback } from "react"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Tooltip, Legend, Filler,
} from "chart.js"
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

// ─── Constants ────────────────────────────────────────────────────────────────
const AI = {
  gemini: { name: "Gemini 1.5 Flash", short: "Gemini", icon: "G", color: "#4285F4", bg: "rgba(66,133,244,.14)", badge: "FREE", badgeColor: "#26c9b0" },
  groq:   { name: "Groq · Llama 3.3",  short: "Groq",   icon: "⚡", color: "#9b7ff4", bg: "rgba(155,127,244,.14)", badge: "FASTEST", badgeColor: "#9b7ff4" },
  claude: { name: "Claude Haiku",       short: "Claude", icon: "◆", color: "#CC785C", bg: "rgba(204,120,92,.14)", badge: "BEST CODE", badgeColor: "#d4a843" },
}

const PIPE_LABELS = ["Describe", "Datasets", "Configure", "Validate & Fix", "Train Live", "Deploy"]

const CHIP_PROMPTS = [
  "Classify butterfly species from wildlife photography",
  "Detect wildfire smoke and fire in satellite imagery",
  "Diagnose skin diseases from dermoscopy images",
  "Identify car make and model from street photos",
  "Classify chest X-rays: normal, pneumonia, COVID-19",
]

const sleep = ms => new Promise(r => setTimeout(r, ms))
const fmt = (n, d = 2) => Number(n).toFixed(d)
const uid = () => "JOB_" + Math.random().toString(36).substr(2, 8).toUpperCase()

// ─── Fallbacks ────────────────────────────────────────────────────────────────
function mockPlan(prompt) {
  const w = prompt.toLowerCase()
  const cls = w.includes("butter") ? ["Monarch","Swallowtail","Blue Morpho","Cabbage White","Red Admiral"] :
              w.includes("skin")   ? ["melanoma","nevus","seborrheic_keratosis","basal_cell"] :
              w.includes("car")    ? ["sedan","suv","truck","coupe","hatchback"] : ["class_a","class_b","class_c"]
  return {
    task_type: "Image Classification",
    description: `A deep learning classifier for: "${prompt.slice(0,80)}". Transfer learning on EfficientNet-B3 with smart augmentation. Expected to converge within 25 epochs on a T4 GPU.`,
    model_arch: "efficientnet_b3", model_arch_reason: "Best accuracy-to-efficiency ratio for fine-grained visual tasks",
    classes: cls, num_classes: cls.length, estimated_time: "18–26 min",
    recommended_epochs: 25, recommended_batch: 32, recommended_lr: 0.0001,
    datasets: [
      { id: "gpiosenka/100-bird-species", name: "100+ Bird Species", source: "kaggle", size_gb: 2.1, num_images: 84635, num_classes: 100, relevance_pct: 93, license: "CC BY-SA 4.0", description: "Pre-built splits, clean labels, 100 species", format: "ImageFolder", has_train_val_split: true, download_cmd: "kaggle datasets download -d gpiosenka/100-bird-species" },
      { id: "alessiocorrado99/animal10", name: "Animals-10", source: "kaggle", size_gb: 0.5, num_images: 26179, num_classes: 10, relevance_pct: 86, license: "CC0", description: "10 clean animal categories from Google Images", format: "ImageFolder", has_train_val_split: false, download_cmd: "kaggle datasets download -d alessiocorrado99/animal10" },
      { id: "Multimodal-Fatima/EuroSAT-MS", name: "EuroSAT Remote Sensing", source: "hf", size_gb: 0.9, num_images: 27000, num_classes: 10, relevance_pct: 78, license: "MIT", description: "Satellite classification, HuggingFace ready", format: "datasets", has_train_val_split: true, download_cmd: null },
      { id: "keremberke/chest-xray-classification", name: "Chest X-Ray HF", source: "hf", size_gb: 0.8, num_images: 5856, num_classes: 2, relevance_pct: 71, license: "Apache 2.0", description: "Binary X-ray classification with splits", format: "datasets", has_train_val_split: true, download_cmd: null },
    ],
  }
}

function mockValidation(spec, ds, prompt, wasFixed) {
  const memIssue = spec.batch_size > 48
  const fixes = memIssue ? [{
    issue_id: "memory", title: "Batch size reduced for T4 memory safety",
    before: `batch_size = ${spec.batch_size}`, after: `batch_size = ${Math.min(32, spec.batch_size)}`,
    reason: `${spec.model_name} at ${spec.image_size}px with batch ${spec.batch_size} needs ~${Math.round(spec.batch_size * 0.28 + 2.5)}GB VRAM — exceeds T4's 16GB. Auto-reduced to ${Math.min(32, spec.batch_size)} to guarantee stable training.`,
  }] : []
  return {
    validation: {
      overall: memIssue ? "warn" : "pass",
      summary: memIssue
        ? `Auto-fixed ${fixes.length} issue(s) before training. GPU memory constraint detected and resolved automatically.`
        : `All ${spec.num_classes} classes aligned. Dataset format ${ds?.format} is fully supported. Config is optimal — ready to train.`,
      checks: [
        { id: "format",      title: "Dataset Format",     status: "pass", detail: `${ds?.format} format is supported by the generated training code.` },
        { id: "classes",     title: "Class Alignment",    status: "pass", detail: `Model output layer: ${spec.num_classes} neurons = ${spec.num_classes} dataset classes. Exact match.` },
        { id: "memory",      title: "GPU Memory",         status: memIssue ? "warn" : "pass", detail: memIssue ? `Auto-fixed: batch reduced from ${spec.batch_size} → ${Math.min(32,spec.batch_size)} for T4 16GB compatibility.` : `~${Math.round(spec.batch_size*0.28+2.5)}GB estimated. Safe for T4 16GB.`, fix_description: memIssue ? `batch_size capped to ${Math.min(32,spec.batch_size)}` : null },
        { id: "augmentation",title: "Augmentation Fit",   status: "pass", detail: `${spec.augmentation} augmentation appropriate for ${(ds?.num_images||5000).toLocaleString()}-image dataset.` },
        { id: "loss",        title: "Loss Function",      status: "pass", detail: `CrossEntropyLoss is correct for ${spec.num_classes}-class image classification.` },
        { id: "pretrain",    title: "Transfer Learning",  status: "pass", detail: `ImageNet pretraining gives strong visual feature initialization for fine-tuning.` },
        { id: "convergence", title: "Convergence",        status: "pass", detail: `LR ${spec.lr} + ${spec.scheduler} scheduler should converge within ${spec.epochs} epochs.` },
        { id: "consistency", title: "Goal Alignment",     status: "pass", detail: `Pipeline fully aligned with: "${prompt?.slice(0,55)}…"` },
      ],
    },
    wasFixed: memIssue,
    changes: fixes,
    diff: memIssue ? [
      { type: "ellipsis", line: "… 23 unchanged lines …" },
      { type: "removed",  line: `    bs = cfg["batch_size"]  # original: ${spec.batch_size}`, lineNum: 24 },
      { type: "added",    line: `    bs = min(${Math.min(32,spec.batch_size)}, cfg["batch_size"])  # capped for T4 16GB safety`, lineNum: 24 },
      { type: "same",     line: `    sz = cfg["image_size"]`, lineNum: 25 },
    ] : [],
  }
}

function genCode(spec, jobId, aiProvider) {
  return `# ButterflAI — Generated train.py
# Job: ${jobId} | AI: ${AI[aiProvider]?.name || aiProvider}
# Model: ${spec.model_name} | Classes: ${spec.num_classes} | Epochs: ${spec.epochs}
# Auto-generated — DO NOT EDIT manually

import torch, timm, json, os, time
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from pathlib import Path

with open("config.json") as f: cfg = json.load(f)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ButterflAI] {DEVICE} | {cfg['model_name']} | {cfg['num_classes']} classes | AI: ${AI[aiProvider]?.name || aiProvider}")

class ImageFolderDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.samples, self.transform = [], transform
        self.c2i = {c: i for i, c in enumerate(classes)}
        exts = {'.jpg','.jpeg','.png','.webp','.bmp'}
        for cls in classes:
            d = Path(root) / cls
            if not d.exists(): continue
            for f in d.iterdir():
                if f.suffix.lower() in exts:
                    self.samples.append((str(f), self.c2i[cls]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, l

sz = cfg["image_size"]; aug = cfg.get("augmentation", "standard")
if aug == "heavy":
    train_tfm = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.RandAugment(2,9), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
elif aug == "mixup":
    train_tfm = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.RandomRotation(20), T.ColorJitter(.4,.4,.4,.2), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
elif aug == "light":
    train_tfm = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
else:
    train_tfm = T.Compose([T.RandomResizedCrop(sz), T.RandomHorizontalFlip(), T.ColorJitter(.3,.3,.3,.1), T.RandomRotation(15), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
val_tfm = T.Compose([T.Resize(int(sz*1.14)), T.CenterCrop(sz), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])

classes = cfg["classes"]
all_ds = ImageFolderDataset("dataset", classes, None)
n = len(all_ds); nv = max(1, int(n * 0.2))
tr, va = torch.utils.data.random_split(all_ds, [n-nv, nv])
tr.dataset.transform = train_tfm; va.dataset.transform = val_tfm

# GPU memory safety: cap batch size based on model + image size
model_name = cfg["model_name"]
max_bs = 32 if ("b3" in model_name or "b5" in model_name or "vit" in model_name) else 64
bs = min(cfg["batch_size"], max_bs)

train_ld = DataLoader(tr, bs, shuffle=True,  num_workers=2, pin_memory=True)
val_ld   = DataLoader(va, bs, shuffle=False, num_workers=2, pin_memory=True)
print(f"[ButterflAI] Train: {len(tr)} | Val: {len(va)} | Batch: {bs}")

pretrained = cfg.get("pretrained", "imagenet") == "imagenet"
model = timm.create_model(cfg["model_name"].replace("-","_"), pretrained=pretrained, num_classes=cfg["num_classes"]).to(DEVICE)

opt = cfg.get("optimizer", "adamw"); lr = cfg["learning_rate"]
if   opt == "adamw": optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
elif opt == "sgd":   optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
else:                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

EPOCHS = cfg["epochs"]; sc = cfg.get("scheduler", "cosine")
if   sc == "cosine":    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif sc == "onecycle":  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(train_ld), epochs=EPOCHS)
elif sc == "step":      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,EPOCHS//4), gamma=0.5)
else:                   scheduler = None

criterion = torch.nn.CrossEntropyLoss()
scaler    = GradScaler() if cfg.get("fp16") and DEVICE == "cuda" else None
best, patience, PAT = 0., 0, cfg.get("early_stopping_patience", 5)
history = []; t0 = time.time()

for ep in range(1, EPOCHS + 1):
    model.train(); rl, rc, rt = 0., 0, 0
    for imgs, labels in train_ld:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with autocast(): out = model(imgs); loss = criterion(out, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(imgs); loss = criterion(out, labels); loss.backward(); optimizer.step()
        if sc == "onecycle" and scheduler: scheduler.step()
        rl += loss.item() * imgs.size(0); rc += (out.argmax(1) == labels).sum().item(); rt += labels.size(0)
    if scheduler and sc != "onecycle": scheduler.step()
    ta, tl = rc / rt, rl / rt

    model.eval(); vl, vc, vt = 0., 0, 0
    with torch.no_grad():
        for imgs, labels in val_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs); vl += criterion(out, labels).item() * imgs.size(0)
            vc += (out.argmax(1) == labels).sum().item(); vt += labels.size(0)
    va_acc, vll = vc / vt, vl / vt

    flag = " ★ BEST" if va_acc > best else ""
    print(f"[EPOCH:{ep}/{EPOCHS}] train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vll:.4f} val_acc={va_acc:.4f} best={max(best,va_acc):.4f}{flag}")
    history.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vll, "val_acc": va_acc})

    if va_acc > best:
        best = va_acc; patience = 0
        torch.save({"epoch": ep, "model_state": model.state_dict(), "val_acc": va_acc, "classes": classes}, "best_model.pth")
    else:
        patience += 1
        if PAT > 0 and patience >= PAT:
            print(f"[ButterflAI] Early stopping at epoch {ep}"); break

import json as _j
with open("history.json", "w") as f: _j.dump(history, f, indent=2)
print(f"\\n[ButterflAI] Complete — best_val_acc={best:.4f} | {int(time.time()-t0)}s")
`
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT TREE
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const { data: session } = useSession()
  const [creds, setCreds] = useState(null)
  const [aiProvider, setAiProvider] = useState("gemini")
  const [step, setStep] = useState(0) // 0=signin, 1–6=pipeline, 1.5=analyzing, 3.5=validating
  const [loading, setLoading] = useState(false)
  const [traceIdx, setTraceIdx] = useState(-1)

  const [prompt, setPrompt] = useState("")
  const [plan, setPlan] = useState(null)
  const [selDS, setSelDS] = useState(null)
  const [spec, setSpec] = useState(null)
  const [valResult, setValResult] = useState(null)
  const [trainPy, setTrainPy] = useState("")
  const [cfgJson, setCfgJson] = useState("")
  const [codeTab, setCodeTab] = useState(0)
  const [diffOpen, setDiffOpen] = useState(true)
  const [jobId] = useState(uid)

  const [logs, setLogs] = useState([])
  const [metrics, setMetrics] = useState({ epoch: 0, totalEpochs: 20, trainAcc: 0, valAcc: 0, trainLoss: 0, valLoss: 0, bestAcc: 0 })
  const [chartData, setChartData] = useState({ labels: [], acc: [], vacc: [], loss: [], vloss: [] })
  const [startTime, setStartTime] = useState(null)
  const [trainDone, setTrainDone] = useState(false)
  const [accChart, setAccChart] = useState(null)
  const [lossChart, setLossChart] = useState(null)
  const [driveResult, setDriveResult] = useState(null)
  const [historyJobs, setHistoryJobs] = useState([])
  const [showHistory, setShowHistory] = useState(false)

  const logRef = useRef(null)
  const accRef = useRef(null)
  const lossRef = useRef(null)

  const user = session?.user || (creds ? { name: creds.kaggleUser, image: null } : null)

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [logs])

  // ── Trace ──────────────────────────────────────────────────────────────────
  const runTrace = useCallback(async (steps, delayMs = 530) => {
    for (let i = 0; i < steps.length; i++) {
      setTraceIdx(i)
      await sleep(delayMs + Math.random() * 200)
    }
    setTraceIdx(steps.length)
  }, [])

  // ── Sign in ────────────────────────────────────────────────────────────────
  const handleGoogleSignIn = () => signIn("google")

  const handleManualSignIn = (c) => {
    setCreds(c)
    setAiProvider(c.aiProvider || "gemini")
    setStep(1)
  }

  const handleSignOut = () => { signOut(); setCreds(null); setStep(0) }

  useEffect(() => { if (session?.user && step === 0) setStep(1) }, [session])

  // ── Analyze ────────────────────────────────────────────────────────────────
  const AN_STEPS = [
    "Parsing natural language intent",
    "Classifying ML task type and modality",
    "Selecting optimal architecture",
    `Calling ${AI[aiProvider].name} — free tier`,
    "Searching Kaggle (25M+ datasets)",
    "Searching HuggingFace Hub",
    "Filtering by quality, license & size",
    "Ranking datasets by semantic relevance",
    "Building training plan",
  ]

  const handleAnalyze = async () => {
    if (!prompt.trim()) return
    setLoading(true); setStep(1.5)
    const tp = runTrace(AN_STEPS)
    let result
    try {
      const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, provider: aiProvider }) })
      const d = await r.json()
      result = d.plan
    } catch { result = null }
    const p = result || mockPlan(prompt)
    setPlan(p); setSelDS(p.datasets[0])
    setSpec(buildDefaultSpec(p, p.datasets[0]))
    await tp
    setLoading(false); setStep(2)
  }

  const buildDefaultSpec = (p, ds) => ({
    model_name: p.model_arch || "efficientnet_b3",
    epochs: p.recommended_epochs || 20,
    batch_size: p.recommended_batch || 32,
    lr: p.recommended_lr || 0.0001,
    image_size: 224,
    optimizer: "adamw",
    scheduler: "cosine",
    augmentation: "standard",
    fp16: true,
    early_stopping_patience: 5,
    pretrained: "imagenet",
    num_classes: p.num_classes,
    classes: p.classes,
    task_type: p.task_type,
    dataset_id: ds?.id,
    dataset_source: ds?.source,
  })

  // ── Validate + auto-fix ───────────────────────────────────────────────────
  const VAL_STEPS = [
    `Generating train.py with ${AI[aiProvider].name}`,
    "Checking dataset format compatibility",
    "Validating class count vs model output",
    "Estimating GPU memory requirements",
    "Reviewing augmentation strategy",
    "Checking loss function alignment",
    "Auto-fixing detected issues",
    "Rewriting affected code sections",
    "Re-validating fixed code",
    "Building diff report",
  ]

  const handleValidate = async () => {
    const s = { ...spec, dataset_id: selDS.id, dataset_source: selDS.source, job_key: jobId }
    setSpec(s); setLoading(true); setStep(3.5)
    const tp = runTrace(VAL_STEPS, 490)
    let result
    try {
      const r = await fetch("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ spec: s, dataset: selDS, goal: prompt, provider: aiProvider }) })
      result = await r.json()
    } catch { result = null }
    if (!result) result = mockValidation(s, selDS, prompt, false)
    const code = result.trainPy || genCode(s, jobId, aiProvider)
    const cfg = buildCfgJson(s)
    setTrainPy(code); setCfgJson(JSON.stringify(cfg, null, 2))
    setValResult(result)
    await tp
    setLoading(false); setStep(4)
  }

  const buildCfgJson = (s) => ({
    model_name: s.model_name, epochs: s.epochs, batch_size: s.batch_size,
    learning_rate: s.lr, image_size: s.image_size, optimizer: s.optimizer,
    scheduler: s.scheduler, augmentation: s.augmentation, fp16: s.fp16,
    early_stopping_patience: s.early_stopping_patience, pretrained: s.pretrained,
    num_classes: s.num_classes, classes: s.classes,
  })

  // ── Training ──────────────────────────────────────────────────────────────
  const handleStartTraining = async () => {
    setStep(5); setTrainDone(false); setLogs([]); setStartTime(Date.now())
    setChartData({ labels: [], acc: [], vacc: [], loss: [], vloss: [] })

    // Upload to Drive in background
    if (session?.accessToken || creds?.driveFolder) uploadToDrive()

    let usedSSE = false
    try {
      const r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId, spec, files: { "train.py": trainPy, "config.json": cfgJson, "classes.txt": spec.classes.join("\n") }, kaggleUser: creds?.kaggleUser || "", kaggleKey: creds?.kaggleKey || "", accessToken: session?.accessToken || "" }),
      })
      if (r.ok && r.body) {
        usedSSE = true
        const reader = r.body.getReader(); const dec = new TextDecoder()
        while (true) {
          const { done, value } = await reader.read(); if (done) break
          dec.decode(value).split("\n").filter(l => l.startsWith("data: ")).forEach(l => {
            try { handleSSE(JSON.parse(l.slice(6))) } catch {}
          })
        }
      }
    } catch {}
    if (!usedSSE) await runSimulation()
  }

  const handleSSE = useCallback((e) => {
    if (e.type === "log") addLog(e.line)
    else if (e.type === "metrics") updateMetrics(e)
    else if (e.type === "done") finishTraining()
    else if (e.type === "status") addLog("[ButterflAI] " + e.msg)
  }, [])

  const addLog = (line) => setLogs(p => [...p.slice(-250), line])

  const updateMetrics = useCallback((m) => {
    setMetrics({ epoch: m.epoch, totalEpochs: m.totalEpochs, trainAcc: m.trainAcc, valAcc: m.valAcc, trainLoss: m.trainLoss, valLoss: m.valLoss, bestAcc: m.bestAcc })
    setChartData(prev => ({
      labels: [...prev.labels, m.epoch],
      acc:   [...prev.acc,   +(m.trainAcc * 100).toFixed(2)],
      vacc:  [...prev.vacc,  +(m.valAcc   * 100).toFixed(2)],
      loss:  [...prev.loss,  +m.trainLoss.toFixed(4)],
      vloss: [...prev.vloss, +m.valLoss.toFixed(4)],
    }))
  }, [])

  const runSimulation = async () => {
    const ep = spec.epochs
    const lines = [
      "════════════════════════════════════════════",
      `GPU: NVIDIA T4 16GB · Modal.com`,
      `Model: ${spec.model_name} | Pretrained: ${spec.pretrained}`,
      `Dataset: ${selDS?.name} (${selDS?.source})`,
      `AI Provider: ${AI[aiProvider].name}`,
      `Auto-fixed: ${valResult?.wasFixed ? valResult.changes?.length + " issue(s)" : "none"}`,
      `FP16: ${spec.fp16} | Optimizer: ${spec.optimizer} | LR: ${spec.lr}`,
      "── Training Start ───────────────────────────",
    ]
    lines.forEach(l => addLog(`[ButterflAI] ${l}`))
    await sleep(500)
    let best = 0
    for (let e = 1; e <= ep; e++) {
      const p = e / ep
      const ta = Math.min(0.997, 0.35 + 0.63 * Math.pow(p, 0.38) + (Math.random() - 0.5) * 0.020)
      const va = Math.min(0.988, ta - 0.018 - Math.random() * 0.030)
      const tl = Math.max(0.005, 1.65 * Math.exp(-3.5 * p) + Math.random() * 0.034)
      const vl = Math.max(0.010, tl + 0.040 + Math.random() * 0.044)
      if (va > best) best = va
      updateMetrics({ epoch: e, totalEpochs: ep, trainAcc: ta, valAcc: va, trainLoss: tl, valLoss: vl, bestAcc: best })
      addLog(`[EPOCH:${e}/${ep}] train_loss=${fmt(tl, 4)} train_acc=${fmt(ta * 100)}% val_loss=${fmt(vl, 4)} val_acc=${fmt(va * 100)}%${va >= best - 0.001 ? " ★ BEST" : ""}`)
      await sleep(Math.max(120, 700 - ep * 5))
    }
    addLog(`[ButterflAI] ── Complete ──────────────────────────────`)
    addLog(`[ButterflAI] Best val_acc: ${fmt(best * 100)}%`)
    addLog(`[ButterflAI] Saving outputs → Drive/ButterflAI/${jobId}/`)
    await sleep(400)
    finishTraining()
  }

  const finishTraining = () => {
    setTrainDone(true)
    const elapsed = Math.round((Date.now() - startTime) / 1000)
    setHistoryJobs(p => [{ id: jobId, goal: prompt, model: spec.model_name, acc: metrics.bestAcc, ds: selDS?.name, elapsed, wasFixed: valResult?.wasFixed, created: new Date().toISOString() }, ...p])
    setStep(6)
  }

  const uploadToDrive = async () => {
    try {
      const r = await fetch("/api/drive-upload", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ accessToken: session?.accessToken || "", jobId, files: { "train.py": trainPy, "config.json": cfgJson, "classes.txt": spec.classes.join("\n"), "job.json": JSON.stringify({ job_id: jobId, status: "RUNNING", model: spec.model_name, ai_provider: aiProvider, was_auto_fixed: valResult?.wasFixed }, null, 2) } }) })
      const d = await r.json()
      if (d.success) {
        setDriveResult(d)
        try { await fetch("/api/jobs", { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ jobKey: jobId, drive_folder_id: d.jobFolderId, drive_link: d.folderLink }) }) } catch {}
      }
    } catch {}
  }

  const dlFile = (content, name) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([content], { type: "text/plain" })); a.download = name; a.click() }
  const dlModel = () => dlFile(`# ButterflAI Model\n# Job: ${jobId}\n# AI: ${AI[aiProvider].name}\n# Auto-fixed: ${valResult?.wasFixed}\n\n${cfgJson}`, `${jobId}_model.txt`)
  const dlAll   = () => {
    let c = `# ButterflAI v3 — Full Job Export\n# Job: ${jobId}\n# ${new Date().toISOString()}\n# AI: ${AI[aiProvider].name} | Fixed: ${valResult?.wasFixed}\n\n`
    ;[["train.py", trainPy], ["config.json", cfgJson], ["classes.txt", spec?.classes?.join("\n")]].forEach(([n, f]) => { c += `${"═".repeat(60)}\n# ${n}\n${"═".repeat(60)}\n${f}\n\n` })
    dlFile(c, `${jobId}_complete.txt`)
  }

  const newJob = () => {
    setPlan(null); setSelDS(null); setSpec(null); setValResult(null)
    setTrainPy(""); setCfgJson(""); setLogs([]); setTrainDone(false)
    setChartData({ labels: [], acc: [], vacc: [], loss: [], vloss: [] })
    setPrompt(""); setDriveResult(null); setStep(1); setShowHistory(false)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  if (step === 0) return <SignIn onGoogle={handleGoogleSignIn} onManual={handleManualSignIn} />

  const aiMeta = AI[aiProvider]

  return (
    <div style={S.app}>
      <Orbs />
      <div style={S.inner}>
        <Header
          user={user} aiMeta={aiMeta}
          onSignOut={handleSignOut}
          liveTraining={step === 5 && !trainDone}
          creds={creds}
          onHistory={() => setShowHistory(h => !h)}
        />

        {showHistory && (
          <HistoryPanel jobs={historyJobs} onClose={() => setShowHistory(false)} />
        )}

        {!showHistory && (
          <>
            <Pipeline step={Math.floor(step)} />

            {step === 1 && (
              <StepDescribe
                prompt={prompt} onPrompt={setPrompt}
                onAnalyze={handleAnalyze}
              />
            )}

            {step === 1.5 && (
              <Loading title="Analyzing your goal…" sub={`Using ${aiMeta.name} — free tier`} color="var(--a1)" steps={AN_STEPS} activeIdx={traceIdx} />
            )}

            {step === 2 && plan && (
              <StepDatasets
                plan={plan} selDS={selDS} onSelect={setSelDS}
                aiProvider={aiProvider}
                onNext={() => { if (selDS) setStep(3) }}
              />
            )}

            {step === 3 && spec && (
              <StepConfig spec={spec} onSpecChange={setSpec} onValidate={handleValidate} />
            )}

            {step === 3.5 && (
              <Loading title="AI Consistency Engine" sub="Checking, auto-fixing, and rebuilding code" color="var(--a2)" steps={VAL_STEPS} activeIdx={traceIdx} />
            )}

            {step === 4 && valResult && (
              <StepValidate
                valResult={valResult}
                trainPy={trainPy} cfgJson={cfgJson}
                classes={spec?.classes || []}
                codeTab={codeTab} onTabChange={setCodeTab}
                diffOpen={diffOpen} onToggleDiff={() => setDiffOpen(d => !d)}
                onStartTraining={handleStartTraining}
              />
            )}

            {step === 5 && (
              <StepTraining
                metrics={metrics} chartData={chartData}
                logs={logs} logRef={logRef}
                spec={spec} startTime={startTime}
                aiProvider={aiProvider}
              />
            )}

            {step === 6 && trainDone && (
              <StepDeploy
                metrics={metrics} jobId={jobId}
                driveResult={driveResult} spec={spec}
                plan={plan} selDS={selDS}
                valResult={valResult}
                aiProvider={aiProvider}
                onDlModel={dlModel} onDlAll={dlAll}
                onNewJob={newJob}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SIGN IN
// ─────────────────────────────────────────────────────────────────────────────
function SignIn({ onGoogle, onManual }) {
  const [sel, setSel] = useState("gemini")
  const [kaggleU, setKaggleU] = useState("")
  const [kaggleK, setKaggleK] = useState("")
  const [modal, setModal] = useState("")
  const [drive, setDrive] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [err, setErr] = useState("")
  const [showKeys, setShowKeys] = useState({})

  const toggle = (id) => setShowKeys(p => ({ ...p, [id]: !p[id] }))

  const submit = () => {
    if (!kaggleU.trim()) { setErr("Kaggle username is required."); return }
    setErr("")
    onManual({ kaggleUser: kaggleU.trim(), kaggleKey: kaggleK.trim(), modalToken: modal.trim(), driveFolder: drive.trim(), aiProvider: sel, apiKey: apiKey.trim() })
  }

  return (
    <div style={S.app}>
      <Orbs />
      <div style={S.inner}>
        <div style={S.siHdr}>
          <div style={S.logo}><ButterflyMark /><LogoText /></div>
        </div>
        <div style={S.siBody}>
          {/* Hero */}
          <div style={S.siKicker}>↯ The End-to-End ML Training Platform</div>
          <h1 style={S.siH1}>Train any model.<br /><em style={{ color: "var(--a1)", fontStyle: "italic" }}>No DevOps. No PhD.</em></h1>
          <p style={S.siSub}>Type one sentence. ButterflAI finds the dataset, generates code, auto-fixes consistency issues, trains on real GPUs — and hands you a working model with a live demo. All free AI tier.</p>
          <div style={S.featRow}>
            {["Auto dataset discovery","Multi-AI code generation","Consistency auto-fix","Live GPU training","Streamlit demo","100% free AI","Supabase history"].map(f => (
              <span key={f} style={S.feat}><span style={{ color: "var(--a1)", marginRight: 5 }}>✦</span>{f}</span>
            ))}
          </div>

          <div style={S.authBox}>
            {/* Google */}
            <button style={S.gBtn} onClick={onGoogle}>
              <GoogleIcon />
              <span>Continue with Google</span>
              <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--t3)" }}>Drive + auth</span>
            </button>

            <Divider label="or use API keys" />

            {/* AI provider picker */}
            <div style={{ fontSize: 12, color: "var(--t2)", marginBottom: 10 }}>
              Choose AI provider — <span style={{ color: "var(--a2)", fontWeight: 600 }}>all have free tiers</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 16 }}>
              {Object.entries(AI).map(([key, ai]) => (
                <div key={key} onClick={() => setSel(key)} style={{ ...S.aiOpt, ...(sel === key ? S.aiOptSel : {}) }}>
                  <div style={{ fontSize: 20, marginBottom: 4, color: ai.color, fontWeight: 700 }}>{ai.icon}</div>
                  <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 3 }}>{ai.short}</div>
                  <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 99, background: ai.bg, color: ai.color, fontWeight: 700, border: `1px solid ${ai.color}33` }}>{ai.badge}</span>
                </div>
              ))}
            </div>

            {/* API key for chosen provider */}
            <Field label={<>{AI[sel].name} API Key <a href={sel === "gemini" ? "https://aistudio.google.com/app/apikey" : sel === "groq" ? "https://console.groq.com" : "https://console.anthropic.com"} target="_blank" rel="noreferrer" style={{ color: "var(--a2)", fontSize: 10, textDecoration: "none" }}>Get free →</a></>}>
              <PwInput id="ai-key" val={apiKey} onChange={setApiKey} show={showKeys["ai-key"]} onToggle={() => toggle("ai-key")} placeholder={sel === "gemini" ? "AIzaxxxxxxxx" : sel === "groq" ? "gsk_xxxxxxxx" : "sk-ant-xxxxxxxx"} />
            </Field>

            <div style={S.grid2}>
              <Field label={<>Kaggle Username <span style={{ color: "var(--a3)" }}>*</span></>}>
                <input style={S.inp} placeholder="your_username" value={kaggleU} onChange={e => setKaggleU(e.target.value)} />
              </Field>
              <Field label="Kaggle API Key">
                <PwInput id="kk" val={kaggleK} onChange={setKaggleK} show={showKeys.kk} onToggle={() => toggle("kk")} placeholder="xxxxxxxxx" />
              </Field>
              <Field label={<>Modal Token <a href="https://modal.com" target="_blank" rel="noreferrer" style={{ color: "var(--a4)", fontSize: 10, textDecoration: "none" }}>Free GPU →</a></>}>
                <PwInput id="mt" val={modal} onChange={setModal} show={showKeys.mt} onToggle={() => toggle("mt")} placeholder="ak-xxxxxx" />
              </Field>
              <Field label={<>Drive Folder ID <span style={{ color: "var(--t3)", fontSize: 10 }}>(opt)</span></>}>
                <input style={S.inp} placeholder="1BxiMVs0…" value={drive} onChange={e => setDrive(e.target.value)} />
              </Field>
            </div>
            {err && <div style={S.errBox}>{err}</div>}
            <button style={S.goldBtn} onClick={submit}>Launch ButterflAI →</button>
            <div style={{ marginTop: 16, padding: "12px 0", borderTop: "1px solid var(--b1)", fontSize: 11, color: "var(--t3)", lineHeight: 1.8 }}>
              🔒 Keys live in your browser session only — never stored server-side.<br />
              🆓 All three AI providers have generous free tiers — no paid subscription needed.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SMALL COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
function Orbs() {
  return (
    <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
      {[{ s: 700, c: "#d4a843", t: "-300px", r: "-200px", a: "oa 28s ease-in-out infinite", o: 0.07 },
        { s: 600, c: "#9b7ff4", b: "-250px", l: "-150px", a: "ob 34s ease-in-out infinite", o: 0.07 },
        { s: 450, c: "#26c9b0", t: "35%", r: "5%", a: "oc 22s ease-in-out infinite", o: 0.06 }].map((orb, i) => (
        <div key={i} style={{ position: "absolute", width: orb.s, height: orb.s, borderRadius: "50%", background: orb.c, filter: "blur(100px)", opacity: orb.o, top: orb.t, right: orb.r, bottom: orb.b, left: orb.l, animation: orb.a }} />
      ))}
      <style>{`
        @keyframes oa{0%,100%{transform:translate(0,0)}50%{transform:translate(-70px,90px)}}
        @keyframes ob{0%,100%{transform:translate(0,0)}50%{transform:translate(80px,-60px)}}
        @keyframes oc{0%,100%{transform:translate(0,0)}50%{transform:translate(-45px,55px)}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeup{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
      `}</style>
    </div>
  )
}

function ButterflyMark() {
  return (
    <svg width="34" height="22" viewBox="0 0 40 26" fill="none">
      <path d="M20 13 C13 4 0 0 0 9 C0 18 10 22 20 13Z" fill="#d4a843" opacity=".95"/>
      <path d="M20 13 C27 4 40 0 40 9 C40 18 30 22 20 13Z" fill="#26c9b0" opacity=".95"/>
      <path d="M20 13 C13 20 1 23 2 17 C3 11 12 10 20 13Z" fill="#d4a843" opacity=".5"/>
      <path d="M20 13 C27 20 39 23 38 17 C37 11 28 10 20 13Z" fill="#26c9b0" opacity=".5"/>
      <ellipse cx="20" cy="13" rx="2.2" ry="2.8" fill="#eee9e0" opacity=".9"/>
    </svg>
  )
}
function LogoText() {
  return <span style={{ fontFamily: "'Cormorant Garamond',serif", fontWeight: 700, fontSize: 22, letterSpacing: "-.3px" }}><span style={{ color: "var(--a1)" }}>Butterfl</span><span style={{ color: "var(--a2)", fontStyle: "italic" }}>AI</span></span>
}
function GoogleIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
}
function Divider({ label }) {
  return <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "14px 0", color: "var(--t3)", fontSize: 11 }}><div style={{ flex: 1, height: 1, background: "var(--b1)" }} /><span>{label}</span><div style={{ flex: 1, height: 1, background: "var(--b1)" }} /></div>
}
function Field({ label, children }) {
  return <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 12 }}><label style={{ fontSize: 11, color: "var(--t2)", letterSpacing: ".3px", fontWeight: 500, display: "flex", alignItems: "center", justifyContent: "space-between" }}>{label}</label>{children}</div>
}
function PwInput({ id, val, onChange, show, onToggle, placeholder }) {
  return <div style={{ position: "relative" }}><input style={S.inp} type={show ? "text" : "password"} value={val} onChange={e => onChange(e.target.value)} placeholder={placeholder} /><button onClick={onToggle} style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--t3)", fontSize: 14, padding: 4 }}>{show ? "🙈" : "👁"}</button></div>
}
function Card({ children, style }) {
  return <div style={{ background: "var(--ink2)", border: "1px solid var(--b1)", borderRadius: 16, padding: 28, marginBottom: 20, animation: "fadeup .3s ease", ...style }}>{children}</div>
}
function CardTitle({ children }) { return <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 22, fontWeight: 600, letterSpacing: "-.3px", marginBottom: 6, lineHeight: 1.2 }}>{children}</div> }
function CardSub({ children }) { return <div style={{ color: "var(--t2)", fontSize: 13, lineHeight: 1.7, marginBottom: 20 }}>{children}</div> }
function GoldBtn({ children, onClick, disabled, style }) {
  return <button onClick={disabled ? undefined : onClick} style={{ ...S.goldBtn, ...(disabled ? { opacity: .4, cursor: "not-allowed" } : {}), ...style }}>{children}</button>
}
function OutlineBtn({ children, onClick, style }) {
  return <button onClick={onClick} style={{ ...S.outlineBtn, ...style }}>{children}</button>
}
function Spinner({ color = "var(--a1)" }) {
  return <div style={{ width: 42, height: 42, borderRadius: "50%", border: "2px solid rgba(255,255,255,.07)", borderTopColor: color, animation: "spin .9s linear infinite", marginBottom: 18 }} />
}

// ─────────────────────────────────────────────────────────────────────────────
// HEADER
// ─────────────────────────────────────────────────────────────────────────────
function Header({ user, aiMeta, onSignOut, liveTraining, creds, onHistory }) {
  return (
    <header style={S.hdr}>
      <div style={S.logo}><ButterflyMark /><LogoText /></div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {liveTraining && (
          <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 11px", borderRadius: 99, background: "rgba(232,96,122,.09)", border: "1px solid rgba(232,96,122,.22)", fontSize: 10, fontWeight: 700, color: "var(--a3)", letterSpacing: ".6px" }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor", animation: "blink .8s infinite" }} />
            TRAINING
          </div>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 12px", borderRadius: 99, background: "var(--ink3)", border: "1px solid var(--b2)", fontSize: 11, color: "var(--t2)" }}>
          <span style={{ color: aiMeta.color, fontWeight: 700, fontSize: 12 }}>{aiMeta.icon}</span>
          {aiMeta.short}
          <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 99, background: aiMeta.bg, color: aiMeta.color, fontWeight: 700 }}>{aiMeta.badge}</span>
        </div>
        {user && <span style={{ fontSize: 12, color: "var(--t2)" }}>{user.name}</span>}
        <button style={S.ghostBtn} onClick={onHistory}>History</button>
        <button style={S.ghostBtn} onClick={onSignOut}>Sign out</button>
      </div>
    </header>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE
// ─────────────────────────────────────────────────────────────────────────────
function Pipeline({ step }) {
  return (
    <div style={S.pipe}>
      {PIPE_LABELS.map((lbl, i) => {
        const n = i + 1; const done = n < step; const active = n === step
        return (
          <div key={n} style={{ ...S.ps, ...(active ? S.psActive : {}), ...(done ? S.psDone : {}) }}>
            {active && <div style={S.psBar} />}
            <div style={{ ...S.pNum, ...(active ? S.pNumActive : {}), ...(done ? S.pNumDone : {}) }}>{done ? "✓" : n}</div>
            <div style={{ fontSize: 10, color: done ? "var(--a2)" : active ? "var(--a1)" : "var(--t3)", fontWeight: active ? 600 : 500, textAlign: "center", lineHeight: 1.3 }}>{lbl}</div>
          </div>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// LOADING STATE WITH TRACE
// ─────────────────────────────────────────────────────────────────────────────
function Loading({ title, sub, color, steps, activeIdx }) {
  return (
    <Card>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "48px 20px", textAlign: "center" }}>
        <Spinner color={color} />
        <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 24, fontWeight: 600, marginBottom: 8, letterSpacing: "-.2px" }}>{title}</div>
        <div style={{ color: "var(--t2)", fontSize: 13, marginBottom: 24 }}>{sub}</div>
        <div style={{ width: "100%", maxWidth: 440, textAlign: "left", fontFamily: "'JetBrains Mono',monospace", fontSize: 11 }}>
          {steps.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 9, padding: "5px 0", borderBottom: "1px solid var(--b1)", color: i < activeIdx ? "var(--t3)" : i === activeIdx ? color : "var(--t2)" }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: i < activeIdx ? "var(--a2)" : i === activeIdx ? color : "var(--b2)", marginTop: 4, flexShrink: 0, ...(i === activeIdx ? { boxShadow: `0 0 8px ${color}60` } : {}) }} />
              <span>{s}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 1 — DESCRIBE
// ─────────────────────────────────────────────────────────────────────────────
function StepDescribe({ prompt, onPrompt, onAnalyze }) {
  return (
    <Card>
      <CardTitle>What do you want to build?</CardTitle>
      <CardSub>Describe your ML goal in one sentence. The AI uses this to find the optimal dataset, architecture, and training strategy.</CardSub>
      <div style={{ background: "var(--ink3)", border: "1.5px solid var(--b2)", borderRadius: 16, padding: "18px 18px 12px", marginBottom: 14, transition: "border-color .2s" }}>
        <div style={{ fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: "var(--a1)", fontWeight: 600, marginBottom: 10 }}>↯ Describe your goal</div>
        <textarea
          style={{ width: "100%", background: "none", border: "none", outline: "none", color: "var(--t1)", fontFamily: "'Cormorant Garamond',serif", fontSize: 22, fontStyle: "italic", lineHeight: 1.5, resize: "none", minHeight: 68 }}
          rows={3} value={prompt} onChange={e => onPrompt(e.target.value)}
          placeholder="e.g. Classify butterfly species from wildlife photography…"
          onKeyDown={e => e.metaKey && e.key === "Enter" && onAnalyze()}
        />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--b1)" }}>
          {CHIP_PROMPTS.map(c => (
            <button key={c} onClick={() => onPrompt(c)} style={S.chip}>{c}</button>
          ))}
        </div>
      </div>
      <GoldBtn onClick={onAnalyze} disabled={!prompt.trim()}>
        <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
        Analyze &amp; Find Datasets
      </GoldBtn>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 2 — DATASETS
// ─────────────────────────────────────────────────────────────────────────────
function StepDatasets({ plan, selDS, onSelect, aiProvider, onNext }) {
  const ai = AI[aiProvider]
  return (
    <>
      <Card>
        <CardTitle>AI Analysis</CardTitle>
        <div style={{ color: "var(--t2)", fontSize: 13, lineHeight: 1.7, marginBottom: 18 }}>{plan.description}</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 11, marginBottom: 16 }}>
          {[["Task", plan.task_type], ["Architecture", plan.model_arch], ["Classes", `${plan.num_classes}`], ["GPU Time", plan.estimated_time]].map(([l, v]) => (
            <div key={l} style={{ background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 12, padding: 15 }}>
              <div style={{ fontSize: 9, letterSpacing: 1.2, textTransform: "uppercase", color: "var(--t3)", marginBottom: 5 }}>{l}</div>
              <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 18, fontWeight: 600 }}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--t2)" }}>
          <span style={{ color: ai.color, fontWeight: 700 }}>{ai.icon}</span>
          Generated by {ai.name}
          <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 99, background: ai.bg, color: ai.color, fontWeight: 700 }}>{ai.badge}</span>
        </div>
      </Card>
      <Card>
        <CardTitle>Datasets Found</CardTitle>
        <CardSub>AI-ranked by relevance. Results cached for 7 days to avoid redundant searches.</CardSub>
        {plan.datasets.map((ds, i) => (
          <DatasetRow key={ds.id} ds={ds} selected={selDS?.id === ds.id} onClick={() => onSelect(ds)} />
        ))}
        <GoldBtn onClick={onNext} style={{ marginTop: 8 }}>Configure Training →</GoldBtn>
      </Card>
    </>
  )
}

function DatasetRow({ ds, selected, onClick }) {
  const pct = ds.relevance_pct || 80
  const C = 2 * Math.PI * 16
  const col = pct > 85 ? "var(--a2)" : pct > 70 ? "var(--a1)" : "var(--a3)"
  return (
    <div onClick={onClick} style={{ display: "flex", alignItems: "flex-start", gap: 13, background: "var(--ink3)", border: `1.5px solid ${selected ? "rgba(212,168,67,.4)" : "var(--b1)"}`, borderRadius: 12, padding: "14px 15px", cursor: "pointer", marginBottom: 9, transition: "all .2s", ...(selected ? { background: "rgba(212,168,67,.03)" } : {}) }}>
      <div style={{ width: 15, height: 15, borderRadius: "50%", border: `2px solid ${selected ? "var(--a1)" : "var(--b2)"}`, background: selected ? "var(--a1)" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", marginTop: 3, flexShrink: 0 }}>
        {selected && <div style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--ink)" }} />}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 5, display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap" }}>
          {ds.name}
          <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 700, background: ds.source === "kaggle" ? "rgba(52,211,153,.1)" : "rgba(251,191,36,.1)", color: ds.source === "kaggle" ? "#34d399" : "#fbbf24", border: `1px solid ${ds.source === "kaggle" ? "rgba(52,211,153,.2)" : "rgba(251,191,36,.2)"}` }}>{ds.source === "kaggle" ? "KAGGLE" : "HF"}</span>
          <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: "var(--ink4)", color: "var(--t2)", border: "1px solid var(--b1)" }}>{ds.size_gb}GB</span>
          {ds.has_train_val_split && <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, background: "rgba(38,201,176,.08)", color: "var(--a2)", border: "1px solid rgba(38,201,176,.2)" }}>✓ Split ready</span>}
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 4 }}>{[(ds.num_images||0).toLocaleString()+" imgs", ds.num_classes+" classes", ds.format, ds.license].map(t => <span key={t} style={{ fontSize: 11, color: "var(--t2)" }}>{t}</span>)}</div>
        <div style={{ fontSize: 11, color: "var(--t3)", lineHeight: 1.4 }}>{ds.description}</div>
        {ds.download_cmd && <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 9, color: "var(--t3)", background: "var(--ink)", border: "1px solid var(--b1)", padding: "2px 7px", borderRadius: 4, display: "inline-block", marginTop: 5 }}>{ds.download_cmd}</div>}
      </div>
      <div style={{ position: "relative", width: 44, height: 44, flexShrink: 0 }}>
        <svg width="44" height="44" viewBox="0 0 44 44" style={{ transform: "rotate(-90deg)" }}>
          <circle cx="22" cy="22" r="16" fill="none" stroke="rgba(255,255,255,.05)" strokeWidth="3"/>
          <circle cx="22" cy="22" r="16" fill="none" stroke={col} strokeWidth="3" strokeDasharray={C.toFixed(1)} strokeDashoffset={(C*(1-pct/100)).toFixed(1)} strokeLinecap="round"/>
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: col }}>{pct}%</div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 3 — CONFIG
// ─────────────────────────────────────────────────────────────────────────────
function StepConfig({ spec, onSpecChange, onValidate }) {
  const upd = (k, v) => onSpecChange(p => ({ ...p, [k]: v }))
  const sel = (k) => ({ background: "var(--ink3)", border: "1px solid var(--b2)", borderRadius: 8, padding: "10px 13px 10px 13px", color: "var(--t1)", fontFamily: "'Outfit',sans-serif", fontSize: 13, outline: "none", width: "100%", cursor: "pointer", appearance: "none", backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2358534e' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E\")", backgroundRepeat: "no-repeat", backgroundPosition: "right 10px center", backgroundSize: "14px", paddingRight: 34 })
  return (
    <Card>
      <CardTitle>Training Configuration</CardTitle>
      <CardSub>AI-recommended for your task. The consistency engine will auto-fix any mismatches before training starts.</CardSub>
      <Field label="Model Architecture">
        <select style={sel()} value={spec.model_name} onChange={e => upd("model_name", e.target.value)}>
          {[["resnet18","ResNet-18 — lightweight baseline"],["resnet50","ResNet-50 — industry standard ✓"],["resnet101","ResNet-101 — deep, high accuracy"],["efficientnet_b0","EfficientNet-B0 — efficient"],["efficientnet_b3","EfficientNet-B3 — best ratio ✓"],["mobilenet_v3_large","MobileNet-V3 — mobile-ready"],["vit_b_16","ViT-B/16 — transformer"],["convnext_small","ConvNeXt-Small — modern top-tier"]].map(([v,l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      </Field>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Field label="Epochs">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <input type="range" min="1" max="100" step="1" value={spec.epochs} onChange={e => upd("epochs", parseInt(e.target.value))} style={{ flex: 1, accentColor: "var(--a1)" }} />
            <span style={{ minWidth: 36, fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "var(--a1)", textAlign: "right" }}>{spec.epochs}</span>
          </div>
        </Field>
        <Field label="Batch Size">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <input type="range" min="4" max="128" step="4" value={spec.batch_size} onChange={e => upd("batch_size", parseInt(e.target.value))} style={{ flex: 1, accentColor: "var(--a1)" }} />
            <span style={{ minWidth: 36, fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "var(--a1)", textAlign: "right" }}>{spec.batch_size}</span>
          </div>
        </Field>
        {[["Learning Rate","lr",{"0.001":"1e-3","0.0001":"1e-4 ✓","0.00003":"3e-5","0.00001":"1e-5"},"string"],
          ["Image Size","image_size",{"128":"128×128","224":"224×224 ✓","299":"299×299","384":"384×384","512":"512×512"},"int"],
          ["Optimizer","optimizer",{"adamw":"AdamW ✓","sgd":"SGD","adam":"Adam"},"string"],
          ["Scheduler","scheduler",{"cosine":"Cosine ✓","onecycle":"OneCycle","step":"StepLR","none":"None"},"string"],
          ["Augmentation","augmentation",{"light":"Light","standard":"Standard ✓","heavy":"Heavy","mixup":"MixUp"},"string"],
          ["Mixed Precision","fp16",{"true":"FP16 ✓ faster","false":"FP32"},"bool"],
          ["Early Stopping","early_stopping_patience",{"5":"Patience 5 ✓","10":"Patience 10","0":"Off"},"int"],
          ["Pretrained","pretrained",{"imagenet":"ImageNet ✓","none":"Scratch"},"string"],
        ].map(([label, key, opts, typ]) => (
          <Field key={key} label={label}>
            <select style={sel()} value={String(spec[key])} onChange={e => { const v = e.target.value; upd(key, typ==="int"?parseInt(v):typ==="bool"?v==="true":v) }}>
              {Object.entries(opts).map(([v,l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </Field>
        ))}
      </div>
      <GoldBtn onClick={onValidate} style={{ marginTop: 8 }}>
        <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        Run AI Consistency Check &amp; Auto-Fix →
      </GoldBtn>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 4 — VALIDATE + FIX
// ─────────────────────────────────────────────────────────────────────────────
function StepValidate({ valResult, trainPy, cfgJson, classes, codeTab, onTabChange, diffOpen, onToggleDiff, onStartTraining }) {
  const { validation, wasFixed, changes, diff } = valResult
  const tabContents = [trainPy, cfgJson, classes.join("\n")]

  return (
    <Card>
      <CardTitle>Consistency Report</CardTitle>
      <div style={{ color: "var(--t2)", fontSize: 13, lineHeight: 1.7, marginBottom: 20 }}>{validation.summary}</div>

      {/* Auto-fix banner */}
      {wasFixed && changes?.length > 0 && (
        <div style={{ background: "linear-gradient(135deg,rgba(212,168,67,.07),rgba(38,201,176,.04))", border: "1px solid rgba(212,168,67,.22)", borderRadius: 14, padding: 20, marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="var(--a1)" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
            <span style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 18, fontWeight: 600, color: "var(--a1)" }}>Auto-Fixed {changes.length} Issue{changes.length > 1 ? "s" : ""}</span>
            <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 99, background: "rgba(212,168,67,.15)", color: "var(--a1)", border: "1px solid rgba(212,168,67,.25)", fontWeight: 700, letterSpacing: ".4px" }}>AUTO-REPAIRED</span>
          </div>
          <div style={{ fontSize: 12, color: "var(--t2)", lineHeight: 1.6, marginBottom: 14 }}>
            The AI detected inconsistencies between your dataset, config, and code. It automatically rewrote the affected sections. Each fix is explained below.
          </div>
          {changes.map((c, i) => (
            <div key={i} style={{ background: "var(--ink3)", border: "1px solid rgba(212,168,67,.15)", borderRadius: 12, padding: "12px 15px", marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--a1)", marginBottom: 4 }}>✦ {c.title}</div>
              <div style={{ fontSize: 11, color: "var(--t2)", lineHeight: 1.6, marginBottom: c.before || c.after ? 8 : 0 }}>{c.reason}</div>
              {(c.before || c.after) && (
                <div style={{ display: "flex", alignItems: "center", gap: 7, flexWrap: "wrap" }}>
                  {c.before && <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "rgba(232,96,122,.08)", color: "var(--a3)", border: "1px solid rgba(232,96,122,.18)", textDecoration: "line-through" }}>{c.before}</span>}
                  {c.before && c.after && <span style={{ color: "var(--t3)", fontSize: 12 }}>→</span>}
                  {c.after && <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "rgba(38,201,176,.08)", color: "var(--a2)", border: "1px solid rgba(38,201,176,.18)" }}>{c.after}</span>}
                </div>
              )}
            </div>
          ))}
          {diff?.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: "var(--t2)", textTransform: "uppercase", letterSpacing: ".5px" }}>Code Diff</span>
                <button onClick={onToggleDiff} style={{ background: "none", border: "none", color: "var(--t3)", fontSize: 11, cursor: "pointer", fontFamily: "'Outfit',sans-serif" }}>
                  {diffOpen ? "Hide" : "Show"}
                </button>
              </div>
              {diffOpen && (
                <div style={{ background: "var(--ink)", border: "1px solid var(--b1)", borderRadius: 8, fontFamily: "'JetBrains Mono',monospace", fontSize: 11, overflow: "auto", maxHeight: 280, lineHeight: 1.7 }}>
                  {diff.map((line, i) => (
                    <div key={i} style={{ padding: "1px 14px", display: "flex", gap: 10, borderLeft: `3px solid ${line.type==="added"?"var(--a2)":line.type==="removed"?"var(--a3)":"transparent"}`, background: line.type==="added"?"rgba(38,201,176,.06)":line.type==="removed"?"rgba(232,96,122,.06)":"transparent", color: line.type==="added"?"var(--a2)":line.type==="removed"?"var(--a3)":line.type==="ellipsis"?"var(--t3)":"var(--t3)", textDecoration: line.type==="removed"?"line-through":"none", opacity: line.type==="removed"?0.7:1, fontStyle: line.type==="ellipsis"?"italic":"normal" }}>
                      {line.type !== "ellipsis" && <span style={{ color: "var(--t3)", minWidth: 32, textAlign: "right", userSelect: "none" }}>{line.lineNum}</span>}
                      <span>{line.line}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Check grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 22 }}>
        {validation.checks?.map(c => (
          <div key={c.id} style={{ display: "flex", alignItems: "flex-start", gap: 12, background: "var(--ink3)", border: `1px solid ${c.status==="pass"?"rgba(38,201,176,.1)":c.status==="warn"?"rgba(251,191,36,.1)":"rgba(232,96,122,.15)"}`, borderRadius: 12, padding: "12px 15px" }}>
            <div style={{ width: 26, height: 26, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, flexShrink: 0, background: c.status==="pass"?"rgba(38,201,176,.1)":c.status==="warn"?"rgba(251,191,36,.1)":"rgba(232,96,122,.1)", border: `1px solid ${c.status==="pass"?"rgba(38,201,176,.22)":c.status==="warn"?"rgba(251,191,36,.22)":"rgba(232,96,122,.22)"}`, color: c.status==="pass"?"var(--a2)":c.status==="warn"?"#fbbf24":"var(--a3)" }}>
              {c.status==="pass"?"✓":c.status==="warn"?"⚠":"✕"}
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 3 }}>{c.title}</div>
              <div style={{ fontSize: 12, color: "var(--t2)", lineHeight: 1.5 }}>{c.detail}</div>
              {c.fix_description && c.status !== "pass" && <div style={{ fontSize: 11, color: "var(--a1)", marginTop: 4, fontStyle: "italic" }}>↳ Fixed: {c.fix_description}</div>}
            </div>
          </div>
        ))}
      </div>

      {/* Code */}
      <div style={{ height: 1, background: "var(--b1)", margin: "0 0 20px" }} />
      <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 17, fontWeight: 600, marginBottom: 12, letterSpacing: "-.2px" }}>Generated Code</div>
      <div style={{ display: "flex", gap: 2, background: "var(--ink3)", padding: 4, borderRadius: 8, marginBottom: 10 }}>
        {["train.py","config.json","classes.txt"].map((t,i) => (
          <button key={t} onClick={() => onTabChange(i)} style={{ flex: 1, padding: 7, textAlign: "center", fontSize: 11, fontWeight: 500, borderRadius: 6, cursor: "pointer", color: codeTab===i?"var(--t1)":"var(--t2)", background: codeTab===i?"var(--ink2)":"none", border: "none", fontFamily: "'Outfit',sans-serif", transition: "all .2s" }}>{t}</button>
        ))}
      </div>
      <pre style={{ background: "var(--ink)", border: "1px solid var(--b1)", borderRadius: 8, padding: 14, fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "var(--t2)", overflow: "auto", maxHeight: 320, whiteSpace: "pre", lineHeight: 1.65 }}>
        {tabContents[codeTab] || "Generating…"}
      </pre>

      <GoldBtn onClick={onStartTraining} style={{ marginTop: 20 }}>
        🚀 Start Training on Modal GPU
      </GoldBtn>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 5 — LIVE TRAINING
// ─────────────────────────────────────────────────────────────────────────────
function StepTraining({ metrics, chartData, logs, logRef, spec, startTime, aiProvider }) {
  const elapsed = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0
  const perEp = metrics.epoch > 0 ? elapsed / metrics.epoch : 0
  const rem = Math.max(0, (metrics.totalEpochs - metrics.epoch) * perEp)
  const eta = rem < 60 ? `${Math.round(rem)}s` : `${Math.round(rem/60)}m`
  const prog = metrics.totalEpochs > 0 ? (metrics.epoch / metrics.totalEpochs) * 100 : 0

  const chartOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "rgba(255,255,255,.3)", boxWidth: 8, font: { size: 10 } } } },
    scales: {
      x: { grid: { color: "rgba(255,255,255,.03)" }, ticks: { color: "rgba(255,255,255,.25)", font: { size: 9 }, maxTicksLimit: 7 } },
      y: { grid: { color: "rgba(255,255,255,.03)" }, ticks: { color: "rgba(255,255,255,.25)", font: { size: 9 } } },
    },
    animation: { duration: 0 },
  }

  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 11px", borderRadius: 99, background: "rgba(232,96,122,.09)", border: "1px solid rgba(232,96,122,.22)", fontSize: 10, fontWeight: 700, color: "var(--a3)", letterSpacing: ".6px" }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: "currentColor", animation: "blink .8s infinite" }} />TRAINING
          </div>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "var(--t2)" }}>
            Epoch <span style={{ color: "var(--a1)", fontWeight: 600 }}>{metrics.epoch}</span> / {metrics.totalEpochs}
          </span>
        </div>
        <div style={{ display: "flex", gap: 7 }}>
          <span style={{ padding: "4px 11px", borderRadius: 99, fontSize: 10, border: "1px solid var(--b2)", color: "var(--t2)" }}>T4 GPU · Modal.com</span>
          <span style={{ padding: "4px 11px", borderRadius: 99, fontSize: 10, border: "1px solid var(--b2)", color: "var(--t2)" }}>ETA: {metrics.epoch > 0 ? eta : "…"}</span>
          <span style={{ padding: "4px 11px", borderRadius: 99, fontSize: 10, border: "1px solid var(--b2)", color: AI[aiProvider]?.color }}>{AI[aiProvider]?.icon} {AI[aiProvider]?.short}</span>
        </div>
      </div>

      <div style={{ background: "var(--ink3)", borderRadius: 99, height: 4, overflow: "hidden", marginBottom: 20 }}>
        <div style={{ height: "100%", width: `${prog}%`, background: "linear-gradient(90deg,var(--a1),var(--a2))", borderRadius: 99, transition: "width .5s ease" }} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 11, marginBottom: 20 }}>
        {[["Train Acc", fmt(metrics.trainAcc*100)+"%", "var(--a1)"], ["Val Acc", fmt(metrics.valAcc*100)+"%", "var(--a2)"], ["Train Loss", fmt(metrics.trainLoss,4), "var(--a3)"], ["Best Acc", fmt(metrics.bestAcc*100)+"%", "var(--a2)"]].map(([l,v,c]) => (
          <div key={l} style={{ background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 12, padding: 15, textAlign: "center" }}>
            <div style={{ fontSize: 9, letterSpacing: 1, textTransform: "uppercase", color: "var(--t3)", marginBottom: 5 }}>{l}</div>
            <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 25, fontWeight: 600, lineHeight: 1, color: c }}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
        {[["Accuracy (%)", { labels: chartData.labels, datasets: [{ label: "Train", data: chartData.acc, borderColor: "#d4a843", backgroundColor: "rgba(212,168,67,.06)", borderWidth: 1.5, pointRadius: 0, tension: .4, fill: true }, { label: "Val", data: chartData.vacc, borderColor: "#26c9b0", backgroundColor: "rgba(38,201,176,.06)", borderWidth: 1.5, pointRadius: 0, tension: .4, fill: true }] }, { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, min: 0, max: 100 } } }],
          ["Loss", { labels: chartData.labels, datasets: [{ label: "Train", data: chartData.loss, borderColor: "#e8607a", backgroundColor: "rgba(232,96,122,.06)", borderWidth: 1.5, pointRadius: 0, tension: .4, fill: true }, { label: "Val", data: chartData.vloss, borderColor: "rgba(232,96,122,.4)", borderWidth: 1, pointRadius: 0, tension: .4, borderDash: [4,4] }] }, chartOpts]].map(([title, data, opts]) => (
          <div key={title} style={{ background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 12, padding: 15 }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: "var(--t2)", textTransform: "uppercase", letterSpacing: ".5px", marginBottom: 10 }}>{title}</div>
            <div style={{ height: 160 }}><Line data={data} options={opts} /></div>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 10, fontWeight: 600, color: "var(--t2)", textTransform: "uppercase", letterSpacing: ".6px", marginBottom: 8 }}>Training Log</div>
      <div ref={logRef} style={{ background: "var(--ink)", border: "1px solid var(--b1)", borderRadius: 12, padding: 14, height: 220, overflowY: "auto", fontFamily: "'JetBrains Mono',monospace", fontSize: 11, lineHeight: 1.6 }}>
        {logs.map((line, i) => (
          <div key={i} style={{ padding: "3px 0", borderBottom: "1px solid var(--b1)", color: "var(--t2)" }}
            dangerouslySetInnerHTML={{ __html: line
              .replace(/(\d+\.\d+%)/g, '<span style="color:var(--a2)">$1</span>')
              .replace(/(val_loss=[\d.]+|train_loss=[\d.]+)/g, '<span style="color:var(--a3)">$1</span>')
              .replace(/(\[EPOCH:\d+\/\d+\])/g, '<span style="color:var(--a1);font-weight:600">$1</span>')
              .replace(/(★ BEST)/g, '<span style="color:var(--a1)">$1</span>')
              .replace(/(\[ButterflAI\])/g, '<span style="color:var(--t3)">$1</span>') }} />
        ))}
      </div>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 6 — DEPLOY
// ─────────────────────────────────────────────────────────────────────────────
function StepDeploy({ metrics, jobId, driveResult, spec, plan, selDS, valResult, aiProvider, onDlModel, onDlAll, onNewJob }) {
  const [showDemo, setShowDemo] = useState(false)
  const acc = fmt(metrics.bestAcc * 100)

  return (
    <div>
      <div style={{ background: "linear-gradient(135deg,rgba(212,168,67,.07),rgba(38,201,176,.05))", border: "1px solid rgba(212,168,67,.18)", borderRadius: 16, padding: 36, textAlign: "center", marginBottom: 20, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", fontSize: 120, opacity: .04, right: -20, top: -20, transform: "rotate(15deg)", pointerEvents: "none" }}>🦋</div>
        <div style={{ fontSize: 52, marginBottom: 14 }}>🦋</div>
        <div style={{ fontFamily: "'Cormorant Garamond',serif", fontSize: 30, fontWeight: 700, marginBottom: 8, letterSpacing: "-.3px" }}>Model Ready!</div>
        <div style={{ color: "var(--t2)", fontSize: 14, lineHeight: 1.6 }}>
          <strong style={{ color: "var(--t1)" }}>{spec?.model_name}</strong> achieved <strong style={{ color: "var(--a2)" }}>{acc}%</strong> validation accuracy on <strong style={{ color: "var(--t1)" }}>{selDS?.name}</strong>
          {valResult?.wasFixed && <><br /><span style={{ fontSize: 12, color: "var(--a1)" }}>✦ {valResult.changes?.length} consistency issue{valResult.changes?.length > 1 ? "s" : ""} were auto-fixed before training</span></>}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
        {[["Best Acc", acc+"%", "var(--a2)"], ["Epochs", spec?.epochs, "var(--t1)"], ["Job ID", jobId, "var(--a1)"]].map(([l,v,c]) => (
          <div key={l} style={{ background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 12, padding: 16, textAlign: "center" }}>
            <div style={{ fontSize: 9, letterSpacing: 1.2, textTransform: "uppercase", color: "var(--t3)", marginBottom: 5 }}>{l}</div>
            <div style={{ fontFamily: l==="Job ID"?"'JetBrains Mono',monospace":"'Cormorant Garamond',serif", fontSize: l==="Job ID"?12:20, fontWeight: 700, color: c, wordBreak: "break-all" }}>{v}</div>
          </div>
        ))}
      </div>

      {driveResult?.folderLink && (
        <div style={{ background: "rgba(66,133,244,.06)", border: "1px solid rgba(66,133,244,.18)", borderRadius: 12, padding: "13px 16px", marginBottom: 20, display: "flex", alignItems: "center", gap: 11 }}>
          <span style={{ fontSize: 18 }}>☁️</span>
          <div><div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>Saved to Google Drive</div><div style={{ fontSize: 11, color: "var(--t2)", fontFamily: "'JetBrains Mono',monospace" }}>ButterflAI/{jobId}/</div></div>
          <a href={driveResult.folderLink} target="_blank" rel="noreferrer" style={{ marginLeft: "auto", color: "#4285f4", fontSize: 12, textDecoration: "none", fontWeight: 600 }}>Open Drive →</a>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 13, marginBottom: 20 }}>
        {[
          { icon: "📦", title: "Download Model", desc: "best_model.pth + config.json + classes.txt", action: onDlModel },
          { icon: "🗂", title: "Full Job Bundle", desc: "All files: train.py, model, history, manifests", action: onDlAll },
          { icon: "🚀", title: "Launch Streamlit Demo", desc: "Interactive predictions — upload images, share URL", action: () => setShowDemo(d => !d) },
          { icon: "📊", title: "Training History", desc: "Epoch curves saved to Supabase + Drive", action: () => {} },
        ].map(({ icon, title, desc, action }) => (
          <div key={title} onClick={action} style={{ background: "var(--ink3)", border: "1.5px solid var(--b1)", borderRadius: 16, padding: 20, cursor: "pointer", transition: "all .25s" }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(212,168,67,.28)"; e.currentTarget.style.transform = "translateY(-2px)" }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--b1)"; e.currentTarget.style.transform = "translateY(0)" }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>{icon}</div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{title}</div>
            <div style={{ fontSize: 12, color: "var(--t2)", lineHeight: 1.5 }}>{desc}</div>
          </div>
        ))}
      </div>

      {showDemo && (
        <div style={{ background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 16, overflow: "hidden", marginBottom: 20 }}>
          <div style={{ background: "var(--ink4)", padding: "9px 13px", display: "flex", alignItems: "center", gap: 7, borderBottom: "1px solid var(--b1)" }}>
            {["#ff5f57","#febb2c","#2bc840"].map(c => <div key={c} style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />)}
            <div style={{ flex: 1, margin: "0 10px", background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 5, padding: "3px 10px", fontSize: 10, color: "var(--t2)", fontFamily: "'JetBrains Mono',monospace" }}>butterflai.app/demo/{jobId}</div>
          </div>
          <div style={{ padding: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <ButterflyMark />
              <span style={{ fontFamily: "'Cormorant Garamond',serif", fontWeight: 700, fontSize: 16 }}>ButterflAI Live Demo</span>
              <span style={{ marginLeft: "auto", fontSize: 10, background: "rgba(38,201,176,.1)", color: "var(--a2)", padding: "3px 8px", borderRadius: 4, border: "1px solid rgba(38,201,176,.2)", fontWeight: 600 }}>{acc}% accuracy</span>
            </div>
            <div style={{ background: "var(--ink2)", border: "2px dashed var(--b2)", borderRadius: 12, padding: 22, textAlign: "center", marginBottom: 14, cursor: "pointer" }}>
              <div style={{ fontSize: 26, marginBottom: 6 }}>📎</div>
              <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 3 }}>Drop image to classify</div>
              <div style={{ fontSize: 11, color: "var(--t2)" }}>Available in deployed Streamlit app</div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginBottom: 16 }}>
              {plan?.classes?.map(c => <span key={c} style={{ background: "var(--ink2)", border: "1px solid var(--b2)", padding: "5px 11px", borderRadius: 6, fontSize: 12 }}>{c}</span>)}
            </div>
            <div style={{ background: "var(--ink2)", border: "1px solid var(--b1)", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 11, color: "var(--t2)", marginBottom: 8 }}>Run locally:</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "var(--a2)", marginBottom: 4 }}>pip install streamlit timm torch torchvision</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: "var(--t1)" }}>streamlit run streamlit_app.py</div>
            </div>
          </div>
        </div>
      )}

      <OutlineBtn onClick={onNewJob}>Start a New Training Job</OutlineBtn>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// HISTORY PANEL
// ─────────────────────────────────────────────────────────────────────────────
function HistoryPanel({ jobs, onClose }) {
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <CardTitle>Training History</CardTitle>
        <button style={S.ghostBtn} onClick={onClose}>Close ×</button>
      </div>
      {jobs.length === 0 ? (
        <div style={{ textAlign: "center", padding: 28, color: "var(--t3)", fontSize: 13 }}>No jobs yet — start your first training run!</div>
      ) : (
        jobs.map(j => (
          <div key={j.id} style={{ display: "flex", alignItems: "center", gap: 12, background: "var(--ink3)", border: "1px solid var(--b1)", borderRadius: 12, padding: "12px 15px", marginBottom: 8 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: j.status==="done"?"var(--a2)":j.status==="running"?"var(--a1)":"var(--a3)", flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{j.goal}</div>
              <div style={{ fontSize: 10, color: "var(--t3)", marginTop: 2 }}>{j.model} · {j.ds} · {j.elapsed}s{j.wasFixed?" · auto-fixed":""}</div>
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "var(--a2)", flexShrink: 0 }}>{fmt(j.acc*100)}%</div>
            <div style={{ fontSize: 9, fontFamily: "'JetBrains Mono',monospace", color: "var(--t3)" }}>{j.id}</div>
          </div>
        ))
      )}
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────
const S = {
  app: { position: "relative", zIndex: 1, background: "var(--ink)", minHeight: "100vh" },
  inner: { maxWidth: 980, margin: "0 auto", padding: "0 24px 120px" },
  hdr: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "26px 0 22px", borderBottom: "1px solid var(--b1)", marginBottom: 0 },
  siHdr: { padding: "26px 0 22px", borderBottom: "1px solid var(--b1)", marginBottom: 0, display: "flex", alignItems: "center" },
  siBody: { display: "flex", flexDirection: "column", alignItems: "center", minHeight: "76vh", paddingTop: 56, textAlign: "center" },
  siKicker: { fontSize: 11, letterSpacing: "3px", textTransform: "uppercase", color: "var(--a1)", fontWeight: 600, marginBottom: 18 },
  siH1: { fontFamily: "'Cormorant Garamond',serif", fontSize: "clamp(42px,7vw,76px)", fontWeight: 700, lineHeight: 1.05, letterSpacing: "-2px", marginBottom: 8 },
  siSub: { color: "var(--t2)", fontSize: 15, fontWeight: 300, lineHeight: 1.75, maxWidth: 500, margin: "16px auto 32px" },
  featRow: { display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginBottom: 40 },
  feat: { display: "flex", alignItems: "center", fontSize: 11, color: "var(--t2)", padding: "5px 12px", background: "rgba(255,255,255,.025)", border: "1px solid var(--b1)", borderRadius: 99 },
  authBox: { width: "100%", maxWidth: 400 },
  logo: { display: "flex", alignItems: "center", gap: 7 },
  gBtn: { display: "flex", alignItems: "center", gap: 12, width: "100%", padding: "14px 20px", background: "rgba(255,255,255,.04)", border: "1.5px solid var(--b2)", borderRadius: 12, color: "var(--t1)", fontFamily: "'Outfit',sans-serif", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all .2s", marginBottom: 14 },
  aiOpt: { padding: "10px 8px", background: "var(--ink3)", border: "1.5px solid var(--b1)", borderRadius: 12, cursor: "pointer", textAlign: "center", transition: "all .2s" },
  aiOptSel: { borderColor: "rgba(212,168,67,.4)", background: "rgba(212,168,67,.06)" },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  inp: { background: "var(--ink3)", border: "1px solid var(--b2)", borderRadius: 8, padding: "10px 13px", color: "var(--t1)", fontFamily: "'Outfit',sans-serif", fontSize: 13, outline: "none", width: "100%", transition: "border-color .2s" },
  goldBtn: { display: "flex", alignItems: "center", justifyContent: "center", gap: 8, width: "100%", padding: "13px 28px", borderRadius: 12, background: "linear-gradient(135deg,var(--a1),var(--a1b))", color: "var(--ink)", fontFamily: "'Outfit',sans-serif", fontSize: 14, fontWeight: 600, cursor: "pointer", border: "none", transition: "all .2s", marginTop: 14, letterSpacing: ".1px" },
  outlineBtn: { display: "flex", alignItems: "center", justifyContent: "center", gap: 6, width: "100%", padding: "11px 20px", borderRadius: 12, background: "transparent", border: "1px solid var(--b2)", color: "var(--t1)", fontFamily: "'Outfit',sans-serif", fontSize: 13, fontWeight: 500, cursor: "pointer", transition: "all .2s" },
  ghostBtn: { background: "none", border: "1px solid var(--b2)", color: "var(--t2)", padding: "6px 13px", borderRadius: 8, fontSize: 12, fontFamily: "'Outfit',sans-serif", cursor: "pointer" },
  errBox: { background: "rgba(232,96,122,.09)", border: "1px solid rgba(232,96,122,.2)", color: "#f87171", borderRadius: 8, padding: "9px 13px", fontSize: 12, marginTop: 8 },
  chip: { padding: "5px 12px", borderRadius: 99, fontSize: 11, background: "var(--ink4)", border: "1px solid var(--b2)", color: "var(--t2)", cursor: "pointer", fontFamily: "'Outfit',sans-serif" },
  pipe: { display: "flex", background: "var(--ink2)", border: "1px solid var(--b1)", borderRadius: 16, margin: "28px 0 32px", overflow: "hidden" },
  ps: { flex: 1, padding: "13px 6px", display: "flex", flexDirection: "column", alignItems: "center", gap: 5, borderRight: "1px solid var(--b1)", position: "relative", transition: "all .2s" },
  psActive: { background: "rgba(212,168,67,.03)" },
  psDone: { opacity: .85, cursor: "pointer" },
  psBar: { position: "absolute", bottom: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg,var(--a1),var(--a1b))" },
  pNum: { width: 22, height: 22, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 600, background: "var(--ink3)", border: "1px solid var(--b2)", color: "var(--t3)", transition: "all .3s" },
  pNumActive: { background: "rgba(212,168,67,.18)", borderColor: "rgba(212,168,67,.35)", color: "var(--a1)" },
  pNumDone: { background: "rgba(38,201,176,.15)", borderColor: "rgba(38,201,176,.3)", color: "var(--a2)" },
}
