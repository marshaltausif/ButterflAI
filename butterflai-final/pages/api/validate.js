// pages/api/validate.js - With Data Intelligence integration
import { callAI, parseJSON, validateAPIKey } from '../../lib/ai'
import { runConsistencyCheck, computeDiff } from '../../lib/consistency'
import { analyzeDataset, generateIntelligenceReport } from '../../lib/data-intelligence'
import { saveConsistencyReport, updateJob } from '../../lib/supabase'

const CODE_SYSTEM = `You are a PyTorch production engineer building for Modal.com T4 GPU.
Output ONLY valid Python code. NO markdown. NO explanation.
Requirements:
- timm for model creation
- Smart Dataset class based on format
- All hyperparams from config.json
- Epoch log format: [EPOCH:X/Y] train_loss=X.XXXX train_acc=X.XXXX val_loss=X.XXXX val_acc=X.XXXX best=X.XXXX
- Save best_model.pth and history.json`

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" })
  }

  const { 
    spec, 
    dataset, 
    goal, 
    jobDbId, 
    provider = "gemini",
    apiKey 
  } = req.body

  // Validate API key if provided
  if (apiKey) {
    const keyValidation = await validateAPIKey(provider, apiKey)
    if (!keyValidation.valid) {
      return res.status(400).json({ error: keyValidation.error })
    }
  }

  try {
    // Step 1: Run Data Intelligence on the dataset
    const intelligence = await analyzeDataset(dataset, apiKey, provider)
    const intelligenceReport = generateIntelligenceReport(intelligence)

    // Check for critical issues
    const criticalIssues = intelligence.issues.filter(i => i.severity === 'critical')
    if (criticalIssues.length > 0) {
      return res.status(400).json({
        error: 'Dataset has critical issues',
        intelligence: intelligenceReport,
        issues: criticalIssues
      })
    }

    // Step 2: Generate train.py code
    const codePrompt = `Config:\n${JSON.stringify(spec, null, 2)}\nDataset: ${JSON.stringify(dataset)}\nData Intelligence: ${JSON.stringify(intelligence)}`
    
    const { text: rawCode, provider: usedProvider } = await callAI(
      CODE_SYSTEM,
      codePrompt,
      provider,
      apiKey
    )
    
    const initialCode = rawCode.replace(/```python|```/gi, "").trim()

    // Step 3: Consistency check + auto-fix
    const result = await runConsistencyCheck({ 
      spec, 
      dataset, 
      goal, 
      trainPy: initialCode, 
      provider,
      apiKey 
    })

    // Step 4: Generate diff
    const diff = result.wasFixed ? computeDiff(result.originalCode, result.trainPy) : []

    // Step 5: Save to Supabase
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
          intelligence: intelligenceReport
        })
        
        if (result.wasFixed) {
          await updateJob(spec.job_key, { 
            was_auto_fixed: true, 
            fix_count: result.changes?.length || 0,
            data_intelligence: intelligenceReport
          })
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
      intelligence: intelligenceReport
    })

  } catch (err) {
    console.error("Validation failed:", err)
    return res.status(500).json({ 
      error: err.message || 'Validation failed' 
    })
  }
}