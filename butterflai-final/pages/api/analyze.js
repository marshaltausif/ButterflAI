// pages/api/analyze.js
import { callAI, parseJSON } from "../../lib/ai"
import { getCachedDatasets, setCachedDatasets } from "../../lib/supabase"
import crypto from "crypto"

const SYSTEM = `You are a world-class ML engineer. Given a natural language ML goal, return ONLY valid JSON (no markdown fences, no extra text):
{
  "task_type": "Image Classification",
  "description": "2-3 sentences describing what will be built and why this architecture",
  "model_arch": "efficientnet_b3",
  "model_arch_reason": "brief reason",
  "framework": "pytorch",
  "classes": ["class_a","class_b"],
  "num_classes": 2,
  "estimated_time": "18-25 min",
  "min_images_needed": 1000,
  "recommended_epochs": 25,
  "recommended_batch": 32,
  "recommended_lr": 0.0001,
  "input_type": "image",
  "datasets": [
    {
      "id": "author/dataset-slug",
      "name": "Full Dataset Name",
      "source": "kaggle",
      "size_gb": 1.2,
      "num_images": 8500,
      "num_classes": 2,
      "relevance_pct": 94,
      "license": "CC0",
      "description": "one sentence about the dataset",
      "format": "ImageFolder",
      "has_train_val_split": true,
      "download_cmd": "kaggle datasets download -d author/dataset-slug"
    }
  ]
}
Rules:
- 4-5 datasets, mix kaggle and hf (source: "hf")
- All IDs must be realistic
- Sort datasets by relevance_pct descending
- model_arch: resnet18|resnet50|resnet101|efficientnet_b0|efficientnet_b3|mobilenet_v3_large|vit_b_16|convnext_small`

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end()
  const { prompt, provider = "gemini" } = req.body
  if (!prompt) return res.status(400).json({ error: "prompt required" })

  // Check cache
  const hash = crypto.createHash("md5").update(prompt.toLowerCase().trim()).digest("hex")
  const cached = await getCachedDatasets(hash)
  if (cached) return res.status(200).json({ plan: cached, cached: true, provider: "cache" })

  try {
    const { text, provider: used } = await callAI(SYSTEM, `ML goal: "${prompt}"`, provider)
    const plan = parseJSON(text)
    await setCachedDatasets(hash, prompt, plan)
    return res.status(200).json({ plan, provider: used })
  } catch (err) {
    console.error("analyze:", err.message)
    return res.status(500).json({ error: err.message })
  }
}
