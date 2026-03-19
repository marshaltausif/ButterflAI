// lib/consistency.js
// The consistency engine: validates, identifies fixes, rewrites code, produces diffs

import { callAI, parseJSON } from "./ai"

// ── Validation system prompt ──────────────────────────────────────────────────
const VALIDATION_SYSTEM = `You are a senior ML engineer performing a consistency audit.
Given a training config, dataset info, and goal, return ONLY JSON:
{
  "overall": "pass|warn|fail",
  "summary": "2 sentence overall assessment",
  "checks": [
    {
      "id": "format",
      "title": "Dataset Format Match",
      "status": "pass|warn|fail",
      "detail": "one sentence explanation",
      "fixable": true,
      "fix_description": "What needs to change to fix this (if status != pass)"
    }
  ]
}
Check IDs must include: format, classes, memory, augmentation, loss, pretrain, convergence, consistency, datasize, imgsize
Status: pass = no issue, warn = suboptimal but trainable, fail = will cause training error`

// ── Code fix system prompt ────────────────────────────────────────────────────
const FIX_SYSTEM = `You are a senior ML engineer fixing a PyTorch training script.
You will receive:
1. The original train.py
2. A list of consistency issues to fix
3. The dataset info and config

Return ONLY a JSON object:
{
  "fixed_code": "complete corrected train.py as a string",
  "changes": [
    {
      "issue_id": "memory",
      "title": "Reduced batch size for GPU memory",
      "before": "batch_size = 64",
      "after": "batch_size = 32",
      "reason": "EfficientNet-B3 at 384px with batch 64 needs ~18GB VRAM, exceeding T4's 16GB limit. Reduced to 32."
    }
  ]
}
Be surgical — only change what's broken. Preserve working code exactly.
Every change MUST have a clear reason explaining why the original was wrong.`

// ── Main: validate + auto-fix ─────────────────────────────────────────────────
export async function runConsistencyCheck({ spec, dataset, goal, trainPy, provider = "gemini" }) {
  // Step 1: Validate
  const valInput = `Dataset: ${JSON.stringify(dataset, null, 2)}
Config: ${JSON.stringify(spec, null, 2)}
Goal: "${goal}"`

  const { text: valRaw, provider: usedProvider } = await callAI(VALIDATION_SYSTEM, valInput, provider)
  const validation = parseJSON(valRaw)

  const failCount = validation.checks.filter(c => c.status === "fail").length
  const warnCount = validation.checks.filter(c => c.status === "warn").length
  const needsFix  = failCount > 0 || warnCount > 0

  if (!needsFix) {
    return {
      validation,
      trainPy,
      fixedCode: null,
      changes: [],
      wasFixed: false,
      provider: usedProvider,
    }
  }

  // Step 2: Auto-fix
  const issuesNeedingFix = validation.checks.filter(c => c.status !== "pass" && c.fixable !== false)

  const fixInput = `ORIGINAL train.py:
\`\`\`python
${trainPy}
\`\`\`

Issues to fix:
${JSON.stringify(issuesNeedingFix, null, 2)}

Dataset info: ${JSON.stringify(dataset)}
Config: ${JSON.stringify(spec)}`

  const { text: fixRaw } = await callAI(FIX_SYSTEM, fixInput, provider)
  let fixResult
  try {
    fixResult = parseJSON(fixRaw)
  } catch {
    // If parsing fails, return original with explanation
    fixResult = {
      fixed_code: trainPy,
      changes: issuesNeedingFix.map(i => ({
        issue_id: i.id,
        title: i.title,
        before: "",
        after: "",
        reason: i.fix_description || i.detail,
      })),
    }
  }

  // Step 3: Re-validate fixed code to confirm issues resolved
  const revalInput = `Dataset: ${JSON.stringify(dataset)}
Config: ${JSON.stringify({ ...spec, _note: "after auto-fix" })}
Goal: "${goal}"
Fixed code excerpt: ${fixResult.fixed_code?.slice(0, 500)}…`

  let revalidation = validation
  try {
    const { text: revalRaw } = await callAI(VALIDATION_SYSTEM, revalInput, provider)
    revalidation = parseJSON(revalRaw)
  } catch { /* use original */ }

  return {
    validation: revalidation,
    trainPy: fixResult.fixed_code || trainPy,
    fixedCode: fixResult.fixed_code,
    changes: fixResult.changes || [],
    wasFixed: true,
    originalCode: trainPy,
    provider: usedProvider,
  }
}

// ── Compute line-level diff for UI display ────────────────────────────────────
export function computeDiff(before, after) {
  if (!before || !after) return []

  const beforeLines = before.split("\n")
  const afterLines  = after.split("\n")

  // Simple unified diff — find changed lines
  const result = []
  const maxLen  = Math.max(beforeLines.length, afterLines.length)

  // Build a map of after lines for quick lookup
  const afterSet = new Set(afterLines)
  const beforeSet = new Set(beforeLines)

  // Use a simple LCS-based approach for short files, line-by-line for long
  let i = 0, j = 0
  while (i < beforeLines.length || j < afterLines.length) {
    const bl = beforeLines[i]
    const al = afterLines[j]

    if (bl === al) {
      result.push({ type: "same", line: bl, lineNum: j + 1 })
      i++; j++
    } else if (bl !== undefined && !afterSet.has(bl)) {
      result.push({ type: "removed", line: bl, lineNum: i + 1 })
      i++
    } else if (al !== undefined && !beforeSet.has(al)) {
      result.push({ type: "added", line: al, lineNum: j + 1 })
      j++
    } else {
      // Both changed — show removed then added
      if (bl !== undefined) { result.push({ type: "removed", line: bl, lineNum: i + 1 }); i++ }
      if (al !== undefined) { result.push({ type: "added",   line: al, lineNum: j + 1 }); j++ }
    }
  }

  // Only return context around changes (±3 lines)
  const changeIndices = new Set(
    result.map((r, i) => r.type !== "same" ? i : -1).filter(i => i >= 0)
  )
  const contextIndices = new Set()
  for (const ci of changeIndices) {
    for (let offset = -3; offset <= 3; offset++) {
      const idx = ci + offset
      if (idx >= 0 && idx < result.length) contextIndices.add(idx)
    }
  }

  // Build final diff with ellipsis between non-context sections
  const filtered = []
  let lastIncluded = -1
  for (let k = 0; k < result.length; k++) {
    if (contextIndices.has(k)) {
      if (lastIncluded >= 0 && k > lastIncluded + 1) {
        filtered.push({ type: "ellipsis", line: `… ${k - lastIncluded - 1} unchanged lines …` })
      }
      filtered.push(result[k])
      lastIncluded = k
    }
  }

  return filtered
}
