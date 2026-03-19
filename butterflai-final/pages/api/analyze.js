// pages/api/analyze.js - REAL analysis with proper validation
import { callAI, parseJSON, validateAPIKey } from '../../lib/ai'
import { searchKaggleDatasets, validateKaggleCredentials } from '../../lib/kaggle'
import { searchHFDatasets } from '../../lib/huggingface'
import { getCachedDatasets, setCachedDatasets } from '../../lib/supabase'
import crypto from 'crypto'

const SYSTEM_PROMPT = `You are a world-class ML engineer. Given a natural language ML goal, return ONLY valid JSON:
{
  "task_type": "Image Classification",
  "description": "2-3 sentences describing what will be built",
  "model_arch": "efficientnet_b3",
  "model_arch_reason": "brief reason",
  "classes": ["class_a", "class_b"],
  "num_classes": 2,
  "estimated_time": "18-25 min",
  "min_images_needed": 1000,
  "recommended_epochs": 25,
  "recommended_batch": 32,
  "recommended_lr": 0.0001,
  "input_type": "image"
}`

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" })
  }

  const { 
    prompt, 
    provider = "gemini", 
    apiKey,
    kaggleUsername,
    kaggleKey,
    searchBoth = true 
  } = req.body

  // Validate inputs
  if (!prompt) {
    return res.status(400).json({ error: "Prompt is required" })
  }

  // Validate API key if provided
  if (apiKey) {
    const keyValidation = await validateAPIKey(provider, apiKey)
    if (!keyValidation.valid) {
      return res.status(400).json({ error: keyValidation.error })
    }
  }

  // Validate Kaggle credentials if searching Kaggle
  if (searchBoth && (!kaggleUsername || !kaggleKey)) {
    return res.status(400).json({ 
      error: "Kaggle username and key are required to search Kaggle datasets. You can disable Kaggle search or provide credentials." 
    })
  }

  // Validate Kaggle credentials
  if (kaggleUsername && kaggleKey) {
    const kaggleValidation = await validateKaggleCredentials(kaggleUsername, kaggleKey)
    if (!kaggleValidation.valid) {
      return res.status(400).json({ error: `Kaggle: ${kaggleValidation.error}` })
    }
  }

  try {
    // Check cache first
    const hash = crypto.createHash('md5')
      .update(prompt.toLowerCase().trim())
      .digest('hex')
    
    const cached = await getCachedDatasets(hash)
    if (cached) {
      return res.status(200).json({ 
        plan: cached, 
        cached: true, 
        provider: 'cache' 
      })
    }

    // Step 1: Get AI analysis of the goal
    const { text: planText, provider: usedProvider } = await callAI(
      SYSTEM_PROMPT,
      `ML goal: "${prompt}"`,
      provider,
      apiKey
    )

    const plan = parseJSON(planText)

    // Step 2: Search for relevant datasets
    const datasets = []

    // Search Kaggle if credentials provided
    if (kaggleUsername && kaggleKey) {
      try {
        const kaggleDatasets = await searchKaggleDatasets(
          prompt, 
          kaggleUsername, 
          kaggleKey,
          5
        )
        datasets.push(...kaggleDatasets)
      } catch (error) {
        console.error('Kaggle search failed:', error)
        // Continue with HF only
      }
    }

    // Search HuggingFace
    try {
      const hfDatasets = await searchHFDatasets(prompt, 5)
      datasets.push(...hfDatasets)
    } catch (error) {
      console.error('HF search failed:', error)
    }

    if (datasets.length === 0) {
      return res.status(404).json({ 
        error: "No datasets found for your query. Try different keywords." 
      })
    }

    // Sort by relevance
    datasets.sort((a, b) => b.relevance_pct - a.relevance_pct)

    // Add datasets to plan
    plan.datasets = datasets.slice(0, 8) // Top 8 datasets

    // Cache the results
    await setCachedDatasets(hash, prompt, plan)

    return res.status(200).json({ 
      plan, 
      provider: usedProvider,
      datasetCount: datasets.length 
    })

  } catch (error) {
    console.error('Analysis failed:', error)
    return res.status(500).json({ 
      error: error.message || 'Failed to analyze goal' 
    })
  }
}