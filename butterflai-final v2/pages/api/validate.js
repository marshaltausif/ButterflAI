// pages/api/validate.js
// REAL: Uses the DatasetProfile from the Modal intelligence layer
// to generate train.py, run consistency checks, and auto-fix issues.
// The profile contains ACTUAL measured data — not guesses.

import { callAI, parseJSON } from "../../lib/ai"
import { runConsistencyCheck, computeDiff } from "../../lib/consistency"
import { saveConsistencyReport, updateJob } from "../../lib/supabase"

// System prompt for code generation — driven by the real DatasetProfile
const CODE_SYSTEM = `You are a senior PyTorch engineer writing production training code for Modal.com T4 GPU.
You will receive a DatasetProfile (real measured data) and a config.
Generate a COMPLETE, RUNNABLE train.py. Output ONLY valid Python. No markdown. No explanation.

The DatasetProfile contains:
- modality: image/tabular/text/audio
- format_analysis: actual file types, image sizes, color modes
- class_analysis: real class counts, imbalance ratios, class weights
- domain_analysis: detected domain and special handling flags
- recommendations: exactly what the code must implement
- preprocessing_spec: exact transforms and loaders to use

Rules:
- Follow recommendations.loader_type exactly
- If recommendations.needs_dicom_loader=true, use pydicom
- If recommendations.use_weighted_sampler=true, add WeightedRandomSampler with class_weights
- If recommendations.input_channels=1, patch first conv layer for grayscale
- Use recommendations.normalize_mean/std (NOT always ImageNet defaults)
- Apply ALL transforms listed in preprocessing_spec.train_transforms
- Print epochs in EXACTLY this format:
  [EPOCH:X/Y] train_loss=X.XXXX train_acc=X.XXXX val_loss=X.XXXX val_acc=X.XXXX best=X.XXXX
- Save best_model.pth and history.json`

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end()

  const {
    spec,
    dataset,
    goal,
    datasetProfile,  // Real DatasetProfile from Modal worker if available
    jobDbId,
    provider = "gemini",
  } = req.body

  try {
    // ── Step 1: Generate train.py ─────────────────────────────────────────────
    // If we have a real DatasetProfile, use it. Otherwise generate without it
    // (will be regenerated on Modal with the real profile anyway).
    const profileContext = datasetProfile
      ? `\nREAL DatasetProfile:\n${JSON.stringify(datasetProfile, null, 2)}`
      : `\nNo profile yet — generate generic code for ${spec.task_type || "image classification"}`

    const codePrompt = `Config:\n${JSON.stringify(spec, null, 2)}
Dataset metadata: ${JSON.stringify(dataset, null, 2)}
Goal: "${goal}"${profileContext}`

    const { text: rawCode, provider: usedProvider } = await callAI(
      CODE_SYSTEM, codePrompt, provider
    )
    const initialCode = rawCode.replace(/```python[\s\S]*?```/g, m =>
      m.replace(/```python\s*/,'').replace(/```\s*/,'')
    ).replace(/```/g,"").trim()

    // ── Step 2: Consistency check + auto-fix using REAL profile data ──────────
    // Pass the real profile as dataset context — checks use actual measured values
    const datasetContext = datasetProfile ? {
      ...dataset,
      _profile: {
        modality: datasetProfile.modality,
        format: datasetProfile.format_analysis?.dominant_ext,
        color_mode: datasetProfile.format_analysis?.dominant_color_mode,
        image_size_p50: datasetProfile.format_analysis?.width_p50,
        total_images: datasetProfile.format_analysis?.total_images,
        imbalance_ratio: datasetProfile.class_analysis?.imbalance_ratio,
        domain: datasetProfile.domain_analysis?.type,
        warnings: datasetProfile.warnings,
        corrupt_count: datasetProfile.format_analysis?.corrupt_count,
        is_grayscale: datasetProfile.format_analysis?.is_grayscale,
        has_dicom: datasetProfile.format_analysis?.has_dicom,
      }
    } : dataset

    const result = await runConsistencyCheck({
      spec,
      dataset: datasetContext,
      goal,
      trainPy: initialCode,
      provider,
    })

    // ── Step 3: Diff ──────────────────────────────────────────────────────────
    const diff = result.wasFixed
      ? computeDiff(result.originalCode, result.trainPy)
      : []

    // ── Step 4: Save to Supabase ──────────────────────────────────────────────
    if (jobDbId) {
      try {
        await saveConsistencyReport(jobDbId, {
          overall:     result.validation.overall,
          summary:     result.validation.summary,
          checks:      result.validation.checks,
          fixes:       result.changes,
          code_before: result.originalCode || initialCode,
          code_after:  result.trainPy,
          was_fixed:   result.wasFixed,
          fix_count:   result.changes?.length || 0,
          fixed_at:    result.wasFixed ? new Date().toISOString() : null,
        })
        if (result.wasFixed) {
          await updateJob(spec.job_key, {
            was_auto_fixed: true,
            fix_count: result.changes?.length || 0,
          })
        }
      } catch (e) {
        console.warn("Supabase save non-critical:", e.message)
      }
    }

    return res.status(200).json({
      trainPy:    result.trainPy,
      validation: result.validation,
      wasFixed:   result.wasFixed,
      changes:    result.changes,
      diff,
      provider:   usedProvider,
    })
  } catch (err) {
    console.error("validate error:", err)
    return res.status(500).json({ error: err.message })
  }
}
