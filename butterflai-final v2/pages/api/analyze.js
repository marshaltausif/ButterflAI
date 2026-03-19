// pages/api/analyze.js
// REAL: Hits actual Kaggle REST API + HuggingFace Hub API
// Claude/Gemini/Groq only used to: extract search terms from goal + rank real results
// NO fake datasets. NO made-up names. Every dataset comes from a real API call.

import crypto from "crypto"
import { callAI, parseJSON } from "../../lib/ai"
import { getCachedDatasets, setCachedDatasets } from "../../lib/supabase"

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end()

  const { prompt, provider = "gemini", kaggleUser, kaggleKey, hfToken } = req.body
  if (!prompt) return res.status(400).json({ error: "prompt required" })

  // Cache check
  const cacheKey = crypto.createHash("md5")
    .update(`${prompt.toLowerCase().trim()}:${kaggleUser || ""}`)
    .digest("hex")
  const cached = await getCachedDatasets(cacheKey)
  if (cached) return res.status(200).json({ plan: cached, cached: true, provider: "cache" })

  // Step 1: AI extracts intent + search terms from goal
  const intentSystem = `You are an ML expert. Given a goal, return ONLY JSON (no fences):
{
  "task_type": "Image Classification",
  "data_modality": "image",
  "search_terms": ["butterflies species", "butterfly classification", "lepidoptera"],
  "classes": ["Monarch","Swallowtail"],
  "num_classes": 2,
  "model_arch": "efficientnet_b3",
  "model_arch_reason": "brief reason",
  "recommended_epochs": 25,
  "recommended_batch": 32,
  "recommended_lr": 0.0001,
  "description": "2-3 sentences about what will be built"
}
data_modality: image | tabular | text | audio | multimodal
search_terms: 3 short keyword phrases for searching Kaggle and HuggingFace
model_arch choices: resnet18|resnet50|resnet101|efficientnet_b0|efficientnet_b3|mobilenet_v3_large|vit_b_16|convnext_small|bert-base-uncased|distilbert-base-uncased|xlm-roberta-base|whisper-small`

  let intent
  try {
    const { text } = await callAI(intentSystem, `ML goal: "${prompt}"`, provider)
    intent = parseJSON(text)
  } catch (e) {
    return res.status(500).json({ error: `Intent parsing failed: ${e.message}` })
  }

  // Step 2: Real parallel search — Kaggle API + HuggingFace API
  const [kaggleResult, hfResult] = await Promise.allSettled([
    searchKaggleReal(intent.search_terms, kaggleUser, kaggleKey),
    searchHuggingFaceReal(intent.search_terms, hfToken),
  ])

  const kaggleDatasets = kaggleResult.status === "fulfilled" ? kaggleResult.value : []
  const hfDatasets     = hfResult.status === "fulfilled"     ? hfResult.value     : []
  const allDatasets    = [...kaggleDatasets, ...hfDatasets]

  const errors = {}
  if (kaggleResult.status === "rejected") errors.kaggle = kaggleResult.reason?.message
  if (hfResult.status === "rejected")     errors.hf     = hfResult.reason?.message

  if (allDatasets.length === 0) {
    return res.status(502).json({
      error: "Could not retrieve any datasets. Check Kaggle credentials and network.",
      errors,
    })
  }

  // Step 3: AI ranks the REAL results by relevance
  const rankSystem = `You are an ML data engineer. Rank these REAL datasets by relevance to the goal.
Return ONLY JSON (no fences):
{
  "ranked": [
    {
      "id": "exact-id-from-input",
      "source": "kaggle",
      "relevance_pct": 94,
      "relevance_reason": "one sentence",
      "warnings": ["large: 8GB", "non-commercial license"]
    }
  ]
}
Only rank datasets from the provided list. Max 6. Highest relevance first.
warnings: note size > 5GB, non-commercial licenses, very few samples, etc.`

  let rankings
  try {
    const { text } = await callAI(
      rankSystem,
      `Goal: "${prompt}"\n\nReal datasets found:\n${JSON.stringify(
        allDatasets.slice(0, 15).map(d => ({
          id: d.id, source: d.source, name: d.name,
          size_gb: d.size_gb, description: d.description,
          tags: d.tags, downloads: d.download_count,
        })),
        null, 2
      )}`,
      provider
    )
    rankings = parseJSON(text)
  } catch {
    rankings = {
      ranked: allDatasets.slice(0, 6).map((d, i) => ({
        id: d.id, source: d.source,
        relevance_pct: Math.max(50, 90 - i * 8),
        relevance_reason: "Matched search terms for your goal",
        warnings: [],
      })),
    }
  }

  // Step 4: Merge rankings back into full metadata
  const metaMap = new Map(allDatasets.map(d => [`${d.source}:${d.id}`, d]))
  const finalDatasets = (rankings.ranked || [])
    .map(r => {
      const meta = metaMap.get(`${r.source}:${r.id}`)
      if (!meta) return null
      return { ...meta, relevance_pct: r.relevance_pct, relevance_reason: r.relevance_reason, warnings: r.warnings || [] }
    })
    .filter(Boolean)

  const plan = {
    task_type: intent.task_type,
    data_modality: intent.data_modality,
    description: intent.description,
    model_arch: intent.model_arch,
    model_arch_reason: intent.model_arch_reason,
    classes: intent.classes,
    num_classes: intent.num_classes,
    recommended_epochs: intent.recommended_epochs,
    recommended_batch: intent.recommended_batch,
    recommended_lr: intent.recommended_lr,
    estimated_time: estimateTime(intent),
    datasets: finalDatasets,
    total_found: allDatasets.length,
    search_terms: intent.search_terms,
    errors: Object.keys(errors).length ? errors : undefined,
  }

  await setCachedDatasets(cacheKey, prompt, plan)
  return res.status(200).json({ plan, provider, real_search: true })
}

// ── REAL Kaggle API ────────────────────────────────────────────────────────────
async function searchKaggleReal(searchTerms, user, key) {
  const u = user || process.env.KAGGLE_USERNAME
  const k = key  || process.env.KAGGLE_KEY
  if (!u || !k) throw new Error("Kaggle credentials missing (KAGGLE_USERNAME / KAGGLE_KEY)")

  const auth = Buffer.from(`${u}:${k}`).toString("base64")
  const seen = new Set()
  const results = []

  for (const term of searchTerms.slice(0, 2)) {
    const url = `https://www.kaggle.com/api/v1/datasets/list?search=${encodeURIComponent(term)}&sortBy=relevance&pageSize=8`
    const res = await fetch(url, {
      headers: { Authorization: `Basic ${auth}`, "Content-Type": "application/json" },
      signal: AbortSignal.timeout(12000),
    })
    if (!res.ok) {
      const body = await res.text()
      throw new Error(`Kaggle ${res.status}: ${body.slice(0, 300)}`)
    }
    const list = await res.json()
    for (const ds of (Array.isArray(list) ? list : [])) {
      if (seen.has(ds.ref)) continue
      seen.add(ds.ref)
      results.push({
        id: ds.ref,
        name: ds.title || ds.ref,
        source: "kaggle",
        url: `https://www.kaggle.com/datasets/${ds.ref}`,
        size_gb: Number(((ds.totalBytes || 0) / 1e9).toFixed(2)),
        size_bytes: ds.totalBytes || 0,
        download_count: ds.downloadCount || 0,
        vote_count: ds.voteCount || 0,
        license: ds.licenseName || "Unknown",
        description: (ds.subtitle || ds.description || "").slice(0, 300),
        tags: (ds.tags || []).map(t => (typeof t === "string" ? t : t.name)).slice(0, 6),
        owner: ds.ownerName || "",
        last_updated: ds.lastUpdated || "",
        download_cmd: `kaggle datasets download -d ${ds.ref} --unzip`,
        num_files: ds.fileCount || ds.totalFiles || 0,
      })
    }
  }
  return results
}

// ── REAL HuggingFace Hub API ───────────────────────────────────────────────────
async function searchHuggingFaceReal(searchTerms, hfToken) {
  const headers = {
    "Content-Type": "application/json",
    ...(hfToken ? { Authorization: `Bearer ${hfToken}` } : {}),
  }
  const seen = new Set()
  const results = []

  for (const term of searchTerms.slice(0, 2)) {
    const url = `https://huggingface.co/api/datasets?search=${encodeURIComponent(term)}&limit=8&full=true`
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(10000) })
    if (!res.ok) throw new Error(`HuggingFace ${res.status}`)
    const list = await res.json()

    for (const ds of (Array.isArray(list) ? list : [])) {
      if (seen.has(ds.id)) continue
      seen.add(ds.id)
      const card = ds.cardData || {}
      const info = card.dataset_info || {}
      const trainSplit = info.splits?.train || {}

      results.push({
        id: ds.id,
        name: card.pretty_name || ds.id.split("/").pop().replace(/[-_]/g, " "),
        source: "hf",
        url: `https://huggingface.co/datasets/${ds.id}`,
        size_gb: Number(((info.dataset_size || 0) / 1e9).toFixed(2)),
        size_bytes: info.dataset_size || 0,
        download_count: ds.downloads || 0,
        vote_count: ds.likes || 0,
        license: card.license || "Unknown",
        description: (ds.description || card.pretty_name || "").slice(0, 300),
        tags: (ds.tags || []).slice(0, 6),
        owner: ds.id.split("/")[0] || "",
        last_updated: ds.lastModified || "",
        hf_load: `load_dataset("${ds.id}")`,
        num_rows: trainSplit.num_examples || 0,
        task_categories: ds.task_categories || [],
        modalities: ds.modalities || [],
        features: info.features ? Object.keys(info.features).slice(0, 10) : [],
      })
    }
  }
  return results
}

function estimateTime(intent) {
  const ep  = intent.recommended_epochs || 20
  const mpm = {
    resnet18: 0.4, resnet50: 0.7, resnet101: 1.1,
    efficientnet_b0: 0.5, efficientnet_b3: 0.9,
    mobilenet_v3_large: 0.4, vit_b_16: 1.4, convnext_small: 0.8,
    "bert-base-uncased": 2.0, "distilbert-base-uncased": 1.2,
    "xlm-roberta-base": 2.5, "whisper-small": 1.8,
  }[intent.model_arch] || 0.8
  return `${Math.round(ep * mpm * 0.8)}–${Math.round(ep * mpm * 1.3)} min`
}
