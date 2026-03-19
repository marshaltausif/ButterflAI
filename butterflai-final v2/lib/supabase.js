// lib/supabase.js — ButterflAI complete Supabase layer
import { createClient } from "@supabase/supabase-js"

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
)

const admin = () =>
  createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  )

// ── Profiles ──────────────────────────────────────────────────────────────────
export async function upsertProfile(profile) {
  const { error } = await admin()
    .from("profiles")
    .upsert(profile, { onConflict: "id" })
  if (error) console.error("upsertProfile:", error.message)
}

export async function getProfile(userId) {
  const { data } = await admin().from("profiles").select("*").eq("id", userId).single()
  return data
}

// ── Jobs ──────────────────────────────────────────────────────────────────────
export async function createJob(userId, jobData) {
  const { data, error } = await admin()
    .from("jobs")
    .insert({ user_id: userId, ...jobData })
    .select()
    .single()
  if (error) throw error
  return data
}

export async function updateJob(jobKey, updates) {
  const { error } = await admin().from("jobs").update({ ...updates, updated_at: new Date().toISOString() }).eq("job_key", jobKey)
  if (error) console.error("updateJob:", error.message)
}

export async function getUserJobs(userId) {
  const { data, error } = await admin()
    .from("jobs")
    .select("*, model_outputs(*)")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(50)
  return { data: data || [], error }
}

export async function getJob(jobKey) {
  const { data } = await admin()
    .from("jobs")
    .select("*, model_outputs(*), consistency_reports(*)")
    .eq("job_key", jobKey)
    .single()
  return data
}

// ── Model outputs + download links ────────────────────────────────────────────
export async function saveModelOutput(jobId, outputs) {
  // outputs: { drive_link, drive_folder_id, model_file_id, streamlit_file_id, files: [...] }
  const { data, error } = await admin()
    .from("model_outputs")
    .insert({ job_id: jobId, ...outputs })
    .select()
    .single()
  if (error) console.error("saveModelOutput:", error.message)
  return data
}

export async function getModelOutputs(jobId) {
  const { data } = await admin()
    .from("model_outputs")
    .select("*")
    .eq("job_id", jobId)
    .order("created_at", { ascending: false })
  return data || []
}

// ── Consistency reports ───────────────────────────────────────────────────────
export async function saveConsistencyReport(jobId, report) {
  const { data, error } = await admin()
    .from("consistency_reports")
    .insert({ job_id: jobId, ...report })
    .select()
    .single()
  if (error) console.error("saveConsistencyReport:", error.message)
  return data
}

// ── Dataset cache ─────────────────────────────────────────────────────────────
export async function getCachedDatasets(queryHash) {
  try {
    const { data } = await admin()
      .from("dataset_cache")
      .select("results, hit_count")
      .eq("query_hash", queryHash)
      .gt("expires_at", new Date().toISOString())
      .single()
    if (data) {
      await admin().from("dataset_cache").update({ hit_count: (data.hit_count || 0) + 1 }).eq("query_hash", queryHash)
      return data.results
    }
  } catch {}
  return null
}

export async function setCachedDatasets(queryHash, query, results) {
  try {
    await admin().from("dataset_cache").upsert({
      query_hash: queryHash, query, results,
      expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
    })
  } catch {}
}

// ── Training epochs ───────────────────────────────────────────────────────────
export async function logEpoch(jobId, epochData) {
  try {
    await admin().from("training_epochs").insert({ job_id: jobId, ...epochData })
  } catch {}
}

export async function getTrainingHistory(jobId) {
  const { data } = await admin()
    .from("training_epochs")
    .select("*")
    .eq("job_id", jobId)
    .order("epoch", { ascending: true })
  return data || []
}
