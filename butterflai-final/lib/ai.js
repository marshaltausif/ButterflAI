// lib/ai.js - REAL AI providers with proper error handling
export const AI_PROVIDERS = {
  gemini: {
    name: "Gemini 1.5 Flash",
    label: "Google Gemini",
    badge: "FREE",
    color: "#4285F4",
    description: "Google's fast free model. 1M token context.",
    icon: "G",
    validateKey: (key) => key?.startsWith('AIza') && key.length > 30,
    keyFormat: "AIza... (should start with 'AIza')"
  },
  groq: {
    name: "Llama 3.3 70B",
    label: "Groq (Llama 3)",
    badge: "FREE · FAST",
    color: "#F55036",
    description: "Meta's Llama 3 on Groq hardware. Blazing fast.",
    icon: "⚡",
    validateKey: (key) => key?.startsWith('gsk_') && key.length > 20,
    keyFormat: "gsk_... (should start with 'gsk_')"
  },
  claude: {
    name: "Claude 3 Haiku",
    label: "Anthropic Claude",
    badge: "FREE TIER",
    color: "#CC785C",
    description: "Anthropic's efficient model. Best for code.",
    icon: "◆",
    validateKey: (key) => key?.startsWith('sk-ant-') && key.length > 30,
    keyFormat: "sk-ant-... (should start with 'sk-ant-')"
  },
}

export async function validateAPIKey(provider, key) {
  if (!key) return { valid: false, error: "API key is required" }
  
  const validator = AI_PROVIDERS[provider]?.validateKey
  if (validator && !validator(key)) {
    return { 
      valid: false, 
      error: `Invalid ${AI_PROVIDERS[provider].name} key format. Expected: ${AI_PROVIDERS[provider].keyFormat}`
    }
  }
  
  // Test the key with a minimal API call
  try {
    const testResult = await testProviderKey(provider, key)
    return testResult
  } catch (error) {
    return { valid: false, error: `Key validation failed: ${error.message}` }
  }
}

async function testProviderKey(provider, key) {
  switch (provider) {
    case 'gemini':
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${key}`,
        { method: 'GET' }
      )
      if (!res.ok) throw new Error('Invalid API key')
      return { valid: true }
      
    case 'groq':
      const groqRes = await fetch('https://api.groq.com/openai/v1/models', {
        headers: { Authorization: `Bearer ${key}` }
      })
      if (!groqRes.ok) throw new Error('Invalid API key')
      return { valid: true }
      
    case 'claude':
      const claudeRes = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json'
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1,
          messages: [{ role: 'user', content: 'test' }]
        })
      })
      if (claudeRes.status === 401) throw new Error('Invalid API key')
      return { valid: true }
      
    default:
      return { valid: false, error: 'Unknown provider' }
  }
}

export async function callAI(systemPrompt, userMessage, provider = "gemini", apiKey = null) {
  // Use provided key or fall back to env var
  const key = apiKey || process.env[`${provider.toUpperCase()}_API_KEY`]
  
  if (!key) {
    throw new Error(`No API key found for ${provider}. Please provide a valid API key.`)
  }

  const order = [provider, ...Object.keys(AI_PROVIDERS).filter(p => p !== provider)]

  for (const p of order) {
    try {
      const result = await callProvider(p, systemPrompt, userMessage, 
        apiKey || process.env[`${p.toUpperCase()}_API_KEY`])
      return { text: result, provider: p }
    } catch (err) {
      console.warn(`AI provider ${p} failed:`, err.message)
      if (p === provider) throw err // Only throw if preferred provider fails
    }
  }
  throw new Error("All AI providers failed")
}

async function callProvider(provider, system, user, apiKey) {
  if (!apiKey) throw new Error(`No API key for ${provider}`)
  
  switch (provider) {
    case "gemini":  return callGemini(system, user, apiKey)
    case "groq":    return callGroq(system, user, apiKey)
    case "claude":  return callClaude(system, user, apiKey)
    default: throw new Error(`Unknown provider: ${provider}`)
  }
}

async function callGemini(system, user, apiKey) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: system }] },
        contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 4096 },
      }),
    }
  )
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.candidates?.[0]?.content?.parts?.[0]?.text || ""
}

async function callGroq(system, user, apiKey) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature: 0.3,
      max_tokens: 4096,
    }),
  })
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.choices?.[0]?.message?.content || ""
}

async function callClaude(system, user, apiKey) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-3-haiku-20240307",
      max_tokens: 4096,
      system,
      messages: [{ role: "user", content: user }],
    }),
  })
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.content?.map(c => c.text || "").join("") || ""
}

export function parseJSON(raw) {
  const clean = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .trim()
  const jsonMatch = clean.match(/\{[\s\S]*\}|\[[\s\S]*\]/)
  if (!jsonMatch) throw new Error("No JSON found in AI response")
  return JSON.parse(jsonMatch[0])
}