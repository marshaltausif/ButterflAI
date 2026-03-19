// pages/api/drive-upload.js
// Uploads all job files to user's Google Drive using their OAuth token

import { google } from "googleapis"
import { Readable } from "stream"

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end()

  const { accessToken, jobId, files, folderName } = req.body
  // files: { "train.py": "...", "config.json": "...", ... }

  if (!accessToken) return res.status(401).json({ error: "No access token" })

  try {
    // Build OAuth2 client from user's session token
    const oauth2Client = new google.auth.OAuth2(
      process.env.GOOGLE_CLIENT_ID,
      process.env.GOOGLE_CLIENT_SECRET
    )
    oauth2Client.setCredentials({ access_token: accessToken })

    const drive = google.drive({ version: "v3", auth: oauth2Client })

    // ── 1. Create root ButterflAI folder if it doesn't exist ──────────────────
    let rootFolderId = await findOrCreateFolder(drive, "ButterflAI", "root")

    // ── 2. Create job subfolder ────────────────────────────────────────────────
    const jobFolderId = await createFolder(drive, jobId, rootFolderId)

    // ── 3. Upload each file ────────────────────────────────────────────────────
    const uploaded = []
    for (const [filename, content] of Object.entries(files)) {
      const mimeType = filename.endsWith(".json")
        ? "application/json"
        : filename.endsWith(".py")
        ? "text/x-python"
        : "text/plain"

      const stream = Readable.from([content])
      const file = await drive.files.create({
        requestBody: {
          name: filename,
          parents: [jobFolderId],
          mimeType,
        },
        media: { mimeType, body: stream },
        fields: "id,name,webViewLink",
      })
      uploaded.push({ name: filename, id: file.data.id, link: file.data.webViewLink })
    }

    // ── 4. Get folder link ─────────────────────────────────────────────────────
    const folderMeta = await drive.files.get({
      fileId: jobFolderId,
      fields: "webViewLink",
    })

    return res.status(200).json({
      success: true,
      jobFolderId,
      folderLink: folderMeta.data.webViewLink,
      uploaded,
      drivePath: `ButterflAI/${jobId}/`,
    })
  } catch (err) {
    console.error("Drive upload error:", err)
    return res.status(500).json({ error: err.message })
  }
}

async function findOrCreateFolder(drive, name, parentId) {
  // Check if folder already exists
  const q = `name='${name}' and mimeType='application/vnd.google-apps.folder' and '${parentId}' in parents and trashed=false`
  const existing = await drive.files.list({ q, fields: "files(id)" })
  if (existing.data.files.length > 0) return existing.data.files[0].id
  return createFolder(drive, name, parentId)
}

async function createFolder(drive, name, parentId) {
  const folder = await drive.files.create({
    requestBody: {
      name,
      mimeType: "application/vnd.google-apps.folder",
      parents: [parentId],
    },
    fields: "id",
  })
  return folder.data.id
}
