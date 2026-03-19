// lib/ai.js
// Multi-provider AI layer — Claude, Gemini Flash (free), Groq Llama 3 (free)
// All providers fall back to the next if they fail

export const AI_PROVIDERS = {
  gemini: {
    name: "Gemini 1.5 Flash",
    label: "Google Gemini",
    badge: "FREE",
    color: "#4285F4",
    description: "Google's fast free model. 1M token context.",
    icon: "G",
  },
  groq: {
    name: "Llama 3.3 70B",
    label: "Groq (Llama 3)",
    badge: "FREE · FAST",
    color: "#F55036",
    description: "Meta's Llama 3 on Groq hardware. Blazing fast.",
    icon: "⚡",
  },
  claude: {
    name: "Claude 3 Haiku",
    label: "Anthropic Claude",
    badge: "FREE TIER",
    color: "#CC785C",
    description: "Anthropic's efficient model. Best for code.",
    icon: "◆",
  },
}

// ── Main call — tries preferred, falls back down the chain ────────────────────
export async function callAI(systemPrompt, userMessage, provider = "gemini") {
  const order = [provider, ...Object.keys(AI_PROVIDERS).filter(p => p !== provider)]

  for (const p of order) {
    try {
      const result = await callProvider(p, systemPrompt, userMessage)
      return { text: result, provider: p }
    } catch (err) {
      console.warn(`AI provider ${p} failed:`, err.message)
      // Try next provider
    }
  }
  throw new Error("All AI providers failed")
}

async function callProvider(provider, system, user) {
  switch (provider) {
    case "gemini":  return callGemini(system, user)
    case "groq":    return callGroq(system, user)
    case "claude":  return callClaude(system, user)
    default: throw new Error(`Unknown provider: ${provider}`)
  }
}

// ── Gemini 1.5 Flash (FREE — 15 RPM, 1M TPM free) ───────────────────────────
async function callGemini(system, user) {
  if (!process.env.GEMINI_API_KEY) throw new Error("No GEMINI_API_KEY")

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: system }] },
        contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 4096,
          responseMimeType: "text/plain",
        },
      }),
    }
  )
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.candidates?.[0]?.content?.parts?.[0]?.text || ""
}

// ── Groq Llama 3.3 70B (FREE — 14400 RPD, 6000 TPM free) ────────────────────
async function callGroq(system, user) {
  if (!process.env.GROQ_API_KEY) throw new Error("No GROQ_API_KEY")

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
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

// ── Anthropic Claude Haiku (lowest cost tier) ────────────────────────────────
async function callClaude(system, user) {
  if (!process.env.ANTHROPIC_API_KEY) throw new Error("No ANTHROPIC_API_KEY")

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 4096,
      system,
      messages: [{ role: "user", content: user }],
    }),
  })
  const data = await res.json()
  if (data.error) throw new Error(data.error.message)
  return data.content?.map(c => c.text || "").join("") || ""
}

// ── Parse JSON safely from any AI response ───────────────────────────────────
export function parseJSON(raw) {
  const clean = raw
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .trim()
  // Sometimes models add explanation before/after JSON
  const jsonMatch = clean.match(/\{[\s\S]*\}|\[[\s\S]*\]/)
  if (!jsonMatch) throw new Error("No JSON found in AI response")
  return JSON.parse(jsonMatch[0])
}
