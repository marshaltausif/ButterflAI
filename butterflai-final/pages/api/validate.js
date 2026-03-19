// pages/api/validate.js — generate code + consistency check + auto-fix
import { callAI, parseJSON } from "../../lib/ai"
import { runConsistencyCheck, computeDiff } from "../../lib/consistency"
import { saveConsistencyReport, updateJob } from "../../lib/supabase"

const CODE_SYSTEM = `You are a PyTorch production engineer building for Modal.com T4 GPU.
Output ONLY valid Python code. NO markdown. NO explanation.
Requirements:
- timm for all model creation (import timm)
- Smart Dataset class: auto-detect ImageFolder vs HuggingFace format from manifest "format" field
- Augmentation: light/standard/heavy/mixup
- Optimizers: adamw/sgd/adam
- Schedulers: cosine/onecycle/step/none
- torch.cuda.amp GradScaler when cfg["fp16"]=true
- All hyperparams from config.json
- Epoch log format (EXACTLY, parser depends on it):
  [EPOCH:X/Y] train_loss=X.XXXX train_acc=X.XXXX val_loss=X.XXXX val_acc=X.XXXX best=X.XXXX
- Save best_model.pth and history.json
- Modal.com compatible`

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end()
  const { spec, dataset, goal, jobDbId, provider = "gemini" } = req.body

  try {
    // 1. Generate train.py
    const { text: rawCode, provider: usedProvider } = await callAI(
      CODE_SYSTEM,
      `Config:\n${JSON.stringify(spec, null, 2)}\nDataset: ${JSON.stringify(dataset)}`,
      provider
    )
    const initialCode = rawCode.replace(/```python|```/gi, "").trim()

    // 2. Consistency check + auto-fix
    const result = await runConsistencyCheck({ spec, dataset, goal, trainPy: initialCode, provider })

    // 3. Diff
    const diff = result.wasFixed ? computeDiff(result.originalCode, result.trainPy) : []

    // 4. Save to Supabase
    if (jobDbId) {
      try {
        await saveConsistencyReport(jobDbId, {
          overall: result.validation.overall,
          summary: result.validation.summary,
          checks: result.validation.checks,
          fixes: result.changes,
          code_before: result.originalCode || initialCode,
          code_after: result.trainPy,
          was_fixed: result.wasFixed,
          fix_count: result.changes?.length || 0,
          fixed_at: result.wasFixed ? new Date().toISOString() : null,
        })
        if (result.wasFixed) {
          await updateJob(spec.job_key, { was_auto_fixed: true, fix_count: result.changes?.length || 0 })
        }
      } catch (e) {
        console.warn("Supabase save skipped:", e.message)
      }
    }

    return res.status(200).json({
      trainPy: result.trainPy,
      validation: result.validation,
      wasFixed: result.wasFixed,
      changes: result.changes,
      diff,
      provider: usedProvider,
    })
  } catch (err) {
    console.error("validate:", err.message)
    return res.status(500).json({ error: err.message })
  }
}
