// pages/api/jobs.js — Job CRUD + model output storage
import {
  createJob, updateJob, getUserJobs, getJob,
  saveModelOutput, getModelOutputs, logEpoch,
} from "../../lib/supabase"

export default async function handler(req, res) {
  const { method } = req

  if (method === "POST") {
    const { action, userId, ...data } = req.body
    if (!userId) return res.status(400).json({ error: "userId required" })

    if (action === "create") {
      try { return res.status(200).json({ job: await createJob(userId, data) }) }
      catch (e) { return res.status(500).json({ error: e.message }) }
    }

    if (action === "save_output") {
      // Called after training completes — saves Drive links + file IDs
      const { jobId, outputs } = data
      try { return res.status(200).json({ output: await saveModelOutput(jobId, outputs) }) }
      catch (e) { return res.status(500).json({ error: e.message }) }
    }

    if (action === "log_epoch") {
      const { jobId, epochData } = data
      await logEpoch(jobId, epochData)
      return res.status(200).json({ ok: true })
    }

    return res.status(400).json({ error: "unknown action" })
  }

  if (method === "GET") {
    const { userId, jobKey } = req.query
    if (jobKey) {
      const job = await getJob(jobKey)
      return res.status(200).json({ job })
    }
    if (userId) {
      const { data, error } = await getUserJobs(userId)
      if (error) return res.status(500).json({ error: error.message })
      return res.status(200).json({ jobs: data })
    }
    return res.status(400).json({ error: "userId or jobKey required" })
  }

  if (method === "PATCH") {
    const { jobKey, ...updates } = req.body
    if (!jobKey) return res.status(400).json({ error: "jobKey required" })
    await updateJob(jobKey, updates)
    return res.status(200).json({ ok: true })
  }

  return res.status(405).end()
}
