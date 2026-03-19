// pages/index.js — ButterflAI v3 — REAL production UI
// Every API call hits a real backend. No simulations. No fake data.

import { useState, useRef, useEffect, useCallback } from "react"
import { useSession, signIn, signOut } from "next-auth/react"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Tooltip, Legend, Filler,
} from "chart.js"
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const AI = {
  gemini: { name:"Gemini 1.5 Flash", icon:"G",  color:"#4285F4", bg:"rgba(66,133,244,.14)",   badge:"FREE"      },
  groq:   { name:"Groq Llama 3.3",   icon:"⚡", color:"#9b7ff4", bg:"rgba(155,127,244,.14)",  badge:"FASTEST"   },
  claude: { name:"Claude Haiku",      icon:"◆",  color:"#CC785C", bg:"rgba(204,120,92,.14)",   badge:"BEST CODE" },
}
const STEPS = ["Describe","Datasets","Configure","Validate & Fix","Train Live","Deploy"]
const uid   = () => "JOB_" + Math.random().toString(36).substr(2,8).toUpperCase()
const sleep = ms => new Promise(r => setTimeout(r, ms))
const fmt   = (n, d=2) => Number(n||0).toFixed(d)

// ── Styles ─────────────────────────────────────────────────────────────────────
const S = {
  page:  { position:"relative", zIndex:1, background:"var(--k)",  minHeight:"100vh" },
  wrap:  { maxWidth:980, margin:"0 auto", padding:"0 24px 120px" },
  hdr:   { display:"flex", alignItems:"center", justifyContent:"space-between",
           padding:"26px 0 22px", borderBottom:"1px solid var(--b1)" },
  logo:  { display:"flex", alignItems:"center", gap:7 },
  logoT: { fontFamily:"var(--fh)", fontSize:22, fontWeight:700, letterSpacing:"-.3px" },
  card:  { background:"var(--k2)", border:"1px solid var(--b1)", borderRadius:16,
           padding:28, marginBottom:20, animation:"fu .3s ease" },
  ct:    { fontFamily:"var(--fh)", fontSize:22, fontWeight:600, letterSpacing:"-.3px",
           marginBottom:6, lineHeight:1.2 },
  cs:    { color:"var(--w2)", fontSize:13, lineHeight:1.7, marginBottom:20 },
  inp:   { background:"var(--k3)", border:"1px solid var(--b2)", borderRadius:8,
           padding:"10px 13px", color:"var(--w1)", fontFamily:"var(--fb)",
           fontSize:13, outline:"none", width:"100%", transition:"border-color .2s" },
  gold:  { display:"flex", alignItems:"center", justifyContent:"center", gap:8,
           width:"100%", padding:"13px 28px", borderRadius:12,
           background:"linear-gradient(135deg,var(--g),var(--gb))",
           color:"var(--k)", fontFamily:"var(--fb)", fontSize:14, fontWeight:700,
           cursor:"pointer", border:"none", transition:"all .2s", marginTop:14 },
  ghost: { background:"none", border:"1px solid var(--b2)", color:"var(--w2)",
           padding:"6px 13px", borderRadius:8, fontSize:12,
           fontFamily:"var(--fb)", cursor:"pointer" },
  pipe:  { display:"flex", background:"var(--k2)", border:"1px solid var(--b1)",
           borderRadius:16, margin:"28px 0 32px", overflow:"hidden" },
  ps:    { flex:1, padding:"13px 6px", display:"flex", flexDirection:"column",
           alignItems:"center", gap:5, borderRight:"1px solid var(--b1)",
           position:"relative", transition:"background .2s" },
  pnum:  { width:22, height:22, borderRadius:"50%", display:"flex",
           alignItems:"center", justifyContent:"center", fontSize:10, fontWeight:600,
           background:"var(--k3)", border:"1px solid var(--b2)", color:"var(--w3)", transition:"all .3s" },
  plbl:  { fontSize:10, color:"var(--w3)", fontWeight:500, textAlign:"center", lineHeight:1.3 },
}

export default function Home() {
  const { data: session } = useSession()
  const [creds,   setCreds]   = useState(null)
  const [ai,      setAi]      = useState("gemini")
  const [step,    setStep]    = useState(0)
  const [loading, setLoading] = useState(false)

  // Pipeline state
  const [prompt,     setPrompt]     = useState("")
  const [plan,       setPlan]       = useState(null)
  const [selDS,      setSelDS]      = useState(null)
  const [spec,       setSpec]       = useState(null)
  const [valResult,  setValResult]  = useState(null)
  const [trainPy,    setTrainPy]    = useState("")
  const [cfgJson,    setCfgJson]    = useState("")
  const [profile,    setProfile]    = useState(null)  // real DatasetProfile
  const [jobId]      = useState(uid)
  const [logs,       setLogs]       = useState([])
  const [metrics,    setMetrics]    = useState({ epoch:0, totalEpochs:20, trainAcc:0,
                                                 valAcc:0, trainLoss:0, valLoss:0, bestAcc:0 })
  const [chart,      setChart]      = useState({ labels:[], acc:[], vacc:[], loss:[], vloss:[] })
  const [startTime,  setStartTime]  = useState(null)
  const [trainDone,  setTrainDone]  = useState(false)
  const [trainError, setTrainError] = useState(null)
  const [codeTab,    setCodeTab]    = useState(0)
  const [jobs,       setJobs]       = useState([])
  const [histOpen,   setHistOpen]   = useState(false)
  const [accChart,   setAccChart]   = useState(null)
  const [lossChart,  setLossChart]  = useState(null)

  const logRef = useRef(null)
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [logs])
  useEffect(() => { if (session?.user && step === 0) setStep(1) }, [session])

  const user = session?.user || (creds ? { name: creds.kaggleUser } : null)

  // ── Auth ──────────────────────────────────────────────────────────────────
  const handleGoogleSignIn = () => signIn("google")
  const handleManualSignIn = c  => { setCreds(c); setAi(c.ai || "gemini"); setStep(1) }
  const handleSignOut      = () => { signOut(); setCreds(null); setStep(0) }

  // ── Step 1: Analyze — REAL Kaggle + HF API calls ──────────────────────────
  const handleAnalyze = async () => {
    if (!prompt.trim()) return
    setLoading(true); setStep(1.5)
    try {
      const r = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          provider: ai,
          kaggleUser: creds?.kaggleUser || "",
          kaggleKey:  creds?.kaggleKey  || "",
          hfToken:    creds?.hfToken    || "",
        }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || "Analyze failed")
      if (!d.plan?.datasets?.length) throw new Error("No datasets found. Check Kaggle credentials.")
      setPlan(d.plan)
      setSelDS(d.plan.datasets[0])
      setSpec(defaultSpec(d.plan, d.plan.datasets[0]))
      setStep(2)
    } catch (e) {
      alert(`Search failed: ${e.message}`)
      setStep(1)
    }
    setLoading(false)
  }

  // ── Step 3 → 4: Validate + consistency check ─────────────────────────────
  const handleValidate = async () => {
    const s = { ...spec, dataset_id: selDS.id, dataset_source: selDS.source, job_key: jobId }
    setSpec(s); setLoading(true); setStep(3.5)
    try {
      const r = await fetch("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec: s, dataset: selDS, goal: prompt,
          datasetProfile: profile,  // real profile if already fetched
          provider: ai,
        }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d.error || "Validation failed")
      setTrainPy(d.trainPy || "")
      setCfgJson(JSON.stringify(buildCfg(s), null, 2))
      setValResult(d)
      setStep(4)
    } catch (e) {
      alert(`Validation failed: ${e.message}`)
      setStep(3)
    }
    setLoading(false)
  }

  // ── Step 5: Training — REAL Modal GPU via SSE ─────────────────────────────
  const handleTrain = async () => {
    setStep(5); setTrainDone(false); setTrainError(null)
    setLogs([]); setStartTime(Date.now())
    setChart({ labels:[], acc:[], vacc:[], loss:[], vloss:[] })
    initCharts()

    const cfg = buildCfg(spec)

    try {
      const r = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId,
          spec: cfg,
          dataset: selDS,
          goal: prompt,
          datasetProfile: profile,
          provider: ai,
          kaggleUser: creds?.kaggleUser || "",
          kaggleKey:  creds?.kaggleKey  || "",
          hfToken:    creds?.hfToken    || "",
          accessToken: session?.accessToken || "",
          driveFolderId: creds?.driveFolder || "",
          files: {
            "train.py":    trainPy,
            "config.json": JSON.stringify(cfg, null, 2),
            "classes.txt": (spec.classes || []).join("\n"),
          },
        }),
      })

      if (!r.ok) {
        const d = await r.json()
        throw new Error(d.error || `HTTP ${r.status}`)
      }

      // Stream SSE
      const reader = r.body.getReader()
      const dec    = new TextDecoder()
      let   buf    = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split("\n")
        buf = lines.pop()
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          try {
            const ev = JSON.parse(line.slice(6))
            if (ev.type === "log")     addLog(ev.line)
            if (ev.type === "status")  addLog(`[ButterflAI] ${ev.msg}`)
            if (ev.type === "metrics") updateMetrics(ev)
            if (ev.type === "error")   { setTrainError(ev.msg); return }
            if (ev.type === "done") {
              setTrainDone(true)
              finishTrain(ev)
              return
            }
          } catch {}
        }
      }
      // SSE ended without explicit done
      setTrainDone(true)
      finishTrain({})

    } catch (e) {
      setTrainError(e.message)
      addLog(`[ButterflAI] ERROR: ${e.message}`)
    }
  }

  const addLog = useCallback(line => {
    setLogs(p => [...p.slice(-500), line])
  }, [])

  const updateMetrics = useCallback(m => {
    setMetrics(m)
    setChart(prev => ({
      labels: [...prev.labels, m.epoch],
      acc:    [...prev.acc,    +(m.trainAcc * 100).toFixed(2)],
      vacc:   [...prev.vacc,   +(m.valAcc   * 100).toFixed(2)],
      loss:   [...prev.loss,   +m.trainLoss.toFixed(4)],
      vloss:  [...prev.vloss,  +m.valLoss.toFixed(4)],
    }))
  }, [])

  const finishTrain = useCallback(ev => {
    const elapsed = Math.round((Date.now() - (startTime || Date.now())) / 1000)
    setJobs(p => [{
      id: jobId, goal: prompt, model: spec?.model_name,
      acc: ev.bestAcc || metrics.bestAcc, ds: selDS?.name,
      elapsed, wasFixed: valResult?.wasFixed, ai,
      driveLink: ev.driveLink, created: new Date().toISOString(),
    }, ...p])
    setStep(6)
  }, [jobId, prompt, spec, selDS, metrics, valResult, ai, startTime])

  // ── Charts ────────────────────────────────────────────────────────────────
  function initCharts() {
    const base = {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ labels:{ color:"rgba(255,255,255,.3)", boxWidth:8, font:{size:10} } } },
      scales:{ x:{ grid:{color:"rgba(255,255,255,.03)"}, ticks:{color:"rgba(255,255,255,.25)",font:{size:9},maxTicksLimit:7} },
               y:{ grid:{color:"rgba(255,255,255,.03)"}, ticks:{color:"rgba(255,255,255,.25)",font:{size:9}} } },
      animation:{duration:0},
    }
    if (accChart)  accChart.destroy()
    if (lossChart) lossChart.destroy()
    setAccChart(null); setLossChart(null)
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function defaultSpec(p, ds) {
    return {
      model_name: p.model_arch || "efficientnet_b3",
      epochs: p.recommended_epochs || 20,
      batch_size: p.recommended_batch || 32,
      lr: p.recommended_lr || 0.0001,
      image_size: 224,
      optimizer: "adamw", scheduler: "cosine",
      augmentation: "standard", fp16: true,
      early_stopping_patience: 5, pretrained: "imagenet",
      num_classes: p.num_classes, classes: p.classes,
      task_type: p.task_type,
      dataset_id: ds?.id, dataset_source: ds?.source,
    }
  }

  function buildCfg(s) {
    return {
      model_name: s.model_name, epochs: s.epochs, batch_size: s.batch_size,
      learning_rate: s.lr, image_size: s.image_size, optimizer: s.optimizer,
      scheduler: s.scheduler, augmentation: s.augmentation, fp16: s.fp16,
      early_stopping_patience: s.early_stopping_patience, pretrained: s.pretrained,
      num_classes: s.num_classes, classes: s.classes,
    }
  }

  function dl(content, name) {
    const a = document.createElement("a")
    a.href = URL.createObjectURL(new Blob([content], { type:"text/plain" }))
    a.download = name; a.click()
  }

  // ── Pipeline nav ──────────────────────────────────────────────────────────
  function Pipeline({ step }) {
    return (
      <div style={S.pipe}>
        {STEPS.map((lbl, i) => {
          const n = i + 1
          const done   = n < Math.floor(step)
          const active = n === Math.floor(step)
          return (
            <div key={n} style={{ ...S.ps,
              ...(active ? { background:"rgba(212,168,67,.04)" } : {}),
              ...(n === STEPS.length ? { borderRight:"none" } : {}),
            }}>
              {active && <div style={{ position:"absolute", bottom:0, left:0, right:0, height:2,
                background:"linear-gradient(90deg,var(--g),var(--gb))" }} />}
              <div style={{ ...S.pnum,
                ...(active ? { background:"rgba(212,168,67,.18)", borderColor:"rgba(212,168,67,.4)", color:"var(--g)" } : {}),
                ...(done   ? { background:"rgba(38,201,176,.15)", borderColor:"rgba(38,201,176,.35)", color:"var(--t)" } : {}),
              }}>
                {done ? "✓" : n}
              </div>
              <div style={{ ...S.plbl,
                ...(active ? { color:"var(--g)", fontWeight:600 } : {}),
                ...(done   ? { color:"var(--t)" } : {}),
              }}>{lbl}</div>
            </div>
          )
        })}
      </div>
    )
  }

  // ── Render ────────────────────────────────────────────────────────────────
  if (!user) return <SignIn onGoogle={handleGoogleSignIn} onManual={handleManualSignIn} ai={ai} onAi={setAi} />

  return (
    <div style={S.page}>
      <Orbs />
      <div style={S.wrap}>
        <header style={S.hdr}>
          <div style={S.logo}>
            <BfMark />
            <div style={S.logoT}><span style={{color:"var(--g)"}}>Butterfl</span><span style={{color:"var(--t)",fontStyle:"italic"}}>AI</span></div>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:9, flexWrap:"wrap" }}>
            {step === 5 && !trainDone && (
              <div style={{ display:"flex", alignItems:"center", gap:5, padding:"4px 11px",
                borderRadius:99, background:"rgba(232,96,122,.09)", border:"1px solid rgba(232,96,122,.22)",
                fontSize:10, fontWeight:700, color:"var(--r)", letterSpacing:".6px" }}>
                <span style={{ width:5, height:5, borderRadius:"50%", background:"currentColor", animation:"blink .8s infinite" }} />
                TRAINING
              </div>
            )}
            <AiChip ai={AI[ai]} />
            {user && <span style={{ fontSize:12, color:"var(--w2)" }}>{user.name}</span>}
            <button style={S.ghost} onClick={() => setHistOpen(h => !h)}>History</button>
            <button style={S.ghost} onClick={handleSignOut}>Sign out</button>
          </div>
        </header>

        {histOpen && (
          <div style={S.card}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
              <div style={S.ct}>Training History</div>
              <button style={S.ghost} onClick={() => setHistOpen(false)}>Close ×</button>
            </div>
            {jobs.length === 0
              ? <div style={{ textAlign:"center", padding:28, color:"var(--w3)", fontSize:13 }}>No jobs yet.</div>
              : jobs.map(j => (
                <div key={j.id} style={{ display:"flex", alignItems:"center", gap:12, background:"var(--k3)",
                  border:"1px solid var(--b1)", borderRadius:12, padding:"12px 15px", marginBottom:8 }}>
                  <div style={{ width:7, height:7, borderRadius:"50%", background:"var(--t)" }} />
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:12, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{j.goal}</div>
                    <div style={{ fontSize:10, color:"var(--w3)", marginTop:2 }}>{j.model} · {j.ds} · {j.elapsed}s</div>
                  </div>
                  <div style={{ fontSize:13, fontWeight:600, color:"var(--t)" }}>{fmt(j.acc * 100)}%</div>
                  {j.driveLink && <a href={j.driveLink} target="_blank" rel="noreferrer" style={{ fontSize:11, color:"#4285f4" }}>Drive →</a>}
                </div>
              ))
            }
          </div>
        )}

        <Pipeline step={step} />

        {/* ── STEP 1: Describe ── */}
        {step === 1 && (
          <div style={S.card}>
            <div style={S.ct}>What do you want to build?</div>
            <div style={S.cs}>Describe your ML goal in one sentence. ButterflAI searches Kaggle + HuggingFace for real datasets and ranks them by semantic relevance.</div>
            <div style={{ background:"var(--k3)", border:"1.5px solid var(--b2)", borderRadius:16, padding:"18px 18px 12px", marginBottom:14 }}>
              <div style={{ fontSize:10, letterSpacing:2, textTransform:"uppercase", color:"var(--g)", fontWeight:600, marginBottom:10 }}>↯ Your goal</div>
              <textarea
                style={{ width:"100%", background:"none", border:"none", outline:"none", color:"var(--w1)",
                  fontFamily:"var(--fh)", fontSize:22, fontStyle:"italic", lineHeight:1.5, resize:"none", minHeight:70 }}
                rows={3} value={prompt} onChange={e => setPrompt(e.target.value)}
                placeholder="e.g. Classify butterfly species from wildlife photography…"
                onKeyDown={e => e.metaKey && e.key === "Enter" && handleAnalyze()}
              />
              <div style={{ display:"flex", flexWrap:"wrap", gap:7, marginTop:12, paddingTop:12, borderTop:"1px solid var(--b1)" }}>
                {["Classify butterfly species from wildlife photography",
                  "Detect wildfire smoke in satellite imagery",
                  "Diagnose skin diseases from dermoscopy images",
                  "Classify chest X-rays: normal vs pneumonia",
                  "Sentiment analysis on customer reviews",
                ].map(c => (
                  <button key={c} onClick={() => setPrompt(c)} style={{ padding:"5px 12px", borderRadius:99,
                    fontSize:11, background:"var(--k4)", border:"1px solid var(--b2)", color:"var(--w2)",
                    cursor:"pointer", fontFamily:"var(--fb)" }}>{c.slice(0,30)}…</button>
                ))}
              </div>
            </div>
            <button style={S.gold} onClick={handleAnalyze} disabled={!prompt.trim()}>
              Search Real Datasets →
            </button>
          </div>
        )}

        {/* ── LOADING ── */}
        {(step === 1.5 || step === 3.5) && (
          <div style={S.card}>
            <div style={{ display:"flex", flexDirection:"column", alignItems:"center", padding:"52px 20px", textAlign:"center" }}>
              <div style={{ width:42, height:42, borderRadius:"50%", border:"2px solid rgba(255,255,255,.07)",
                borderTopColor: step === 1.5 ? "var(--g)" : "var(--t)",
                animation:"spin .9s linear infinite", marginBottom:18 }} />
              <div style={{ fontFamily:"var(--fh)", fontSize:24, fontWeight:600, marginBottom:8 }}>
                {step === 1.5 ? "Searching Kaggle + HuggingFace…" : "Running Consistency Check…"}
              </div>
              <div style={{ color:"var(--w2)", fontSize:13 }}>
                {step === 1.5 ? `Calling real APIs with ${AI[ai].name}` : "Validating + auto-fixing code"}
              </div>
            </div>
          </div>
        )}

        {/* ── STEP 2: Datasets ── */}
        {step === 2 && plan && (
          <>
            <div style={S.card}>
              <div style={S.ct}>AI Analysis</div>
              <div style={{ color:"var(--w2)", fontSize:13, lineHeight:1.7, marginBottom:18 }}>{plan.description}</div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:11, marginBottom:16 }}>
                {[["Task", plan.task_type], ["Architecture", plan.model_arch], ["Classes", plan.num_classes], ["GPU Time", plan.estimated_time]].map(([l,v]) => (
                  <div key={l} style={{ background:"var(--k3)", border:"1px solid var(--b1)", borderRadius:12, padding:15 }}>
                    <div style={{ fontSize:9, letterSpacing:1.2, textTransform:"uppercase", color:"var(--w3)", marginBottom:5 }}>{l}</div>
                    <div style={{ fontFamily:"var(--fh)", fontSize:18, fontWeight:600 }}>{String(v)}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize:12, color:"var(--w2)" }}>
                Found <strong style={{ color:"var(--w1)" }}>{plan.total_found}</strong> real datasets across Kaggle + HuggingFace.
                Searched: <em style={{ color:"var(--w3)" }}>{plan.search_terms?.join(", ")}</em>
                {plan.errors && <span style={{ color:"var(--r)", marginLeft:8 }}>
                  {Object.entries(plan.errors).map(([k,v]) => `${k}: ${v}`).join("; ")}
                </span>}
              </div>
            </div>

            <div style={S.card}>
              <div style={S.ct}>Real Datasets Found</div>
              <div style={S.cs}>Ranked by semantic relevance to your goal. Cached 7 days in Supabase.</div>
              {plan.datasets.map((ds, i) => (
                <DatasetRow key={ds.id} ds={ds} selected={selDS?.id === ds.id}
                  onClick={() => { setSelDS(ds); setSpec(defaultSpec(plan, ds)) }} />
              ))}
              <button style={{ ...S.gold, marginTop:8 }} onClick={() => setStep(3)}>Configure Training →</button>
            </div>
          </>
        )}

        {/* ── STEP 3: Config ── */}
        {step === 3 && spec && (
          <div style={S.card}>
            <div style={S.ct}>Training Configuration</div>
            <div style={S.cs}>AI-prefilled for your task. The consistency engine checks 10 dimensions before training starts.</div>
            <ConfigPanel spec={spec} onSpec={setSpec} />
            <button style={S.gold} onClick={handleValidate}>Run AI Consistency Check &amp; Generate Code →</button>
          </div>
        )}

        {/* ── STEP 4: Validate ── */}
        {step === 4 && valResult && (
          <div style={S.card}>
            <div style={S.ct}>Consistency Report</div>
            <div style={{ color:"var(--w2)", fontSize:13, lineHeight:1.7, marginBottom:20 }}>
              {valResult.validation?.summary}
            </div>
            <ConsistencyBanner valResult={valResult} />
            <CheckGrid checks={valResult.validation?.checks || []} />
            <div style={{ height:1, background:"var(--b1)", margin:"20px 0" }} />
            <CodeTabs trainPy={trainPy} cfgJson={cfgJson} classes={(spec.classes||[]).join("\n")}
              tab={codeTab} onTab={setCodeTab} />
            {profile && <ProfileSummary profile={profile} />}
            <button style={{ ...S.gold, marginTop:20 }} onClick={handleTrain}>🚀 Start Training on Modal GPU</button>
          </div>
        )}

        {/* ── STEP 5: Training ── */}
        {step === 5 && (
          <div style={S.card}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16, flexWrap:"wrap", gap:10 }}>
              <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                {!trainDone && !trainError && (
                  <div style={{ display:"flex", alignItems:"center", gap:5, padding:"4px 11px", borderRadius:99,
                    background:"rgba(232,96,122,.09)", border:"1px solid rgba(232,96,122,.22)",
                    fontSize:10, fontWeight:700, color:"var(--r)" }}>
                    <span style={{ width:5, height:5, borderRadius:"50%", background:"currentColor", animation:"blink .8s infinite" }} />LIVE
                  </div>
                )}
                {trainDone && <div style={{ color:"var(--t)", fontWeight:600, fontSize:13 }}>✓ Complete</div>}
                {trainError && <div style={{ color:"var(--r)", fontSize:13 }}>✗ {trainError}</div>}
                <span style={{ fontFamily:"var(--fm)", fontSize:12, color:"var(--w2)" }}>
                  Epoch <span style={{ color:"var(--g)", fontWeight:600 }}>{metrics.epoch}</span> / {metrics.totalEpochs}
                </span>
              </div>
              <div style={{ display:"flex", gap:7 }}>
                <span style={{ padding:"4px 11px", borderRadius:99, fontSize:10, border:"1px solid var(--b2)", color:"var(--w2)" }}>T4 · Modal.com</span>
                <span style={{ padding:"4px 11px", borderRadius:99, fontSize:10, border:"1px solid rgba(212,168,67,.3)", color:"var(--g)" }}>{AI[ai].icon} {AI[ai].name}</span>
              </div>
            </div>

            <div style={{ background:"var(--k3)", borderRadius:99, height:4, overflow:"hidden", marginBottom:20 }}>
              <div style={{ height:"100%", width:`${metrics.totalEpochs > 0 ? (metrics.epoch/metrics.totalEpochs*100) : 0}%`,
                background:"linear-gradient(90deg,var(--g),var(--t))", borderRadius:99, transition:"width .4s ease" }} />
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:11, marginBottom:20 }}>
              {[["Train Acc", fmt(metrics.trainAcc*100)+"%", "var(--g)"],
                ["Val Acc",   fmt(metrics.valAcc*100)+"%",   "var(--t)"],
                ["Train Loss",fmt(metrics.trainLoss,4),      "var(--r)"],
                ["Best Acc",  fmt(metrics.bestAcc*100)+"%",  "var(--t)"]].map(([l,v,c]) => (
                <div key={l} style={{ background:"var(--k3)", border:"1px solid var(--b1)", borderRadius:12, padding:15, textAlign:"center" }}>
                  <div style={{ fontSize:9, letterSpacing:1, textTransform:"uppercase", color:"var(--w3)", marginBottom:5 }}>{l}</div>
                  <div style={{ fontFamily:"var(--fh)", fontSize:25, fontWeight:700, lineHeight:1, color:c }}>{v}</div>
                </div>
              ))}
            </div>

            <LiveCharts chart={chart} />

            <div style={{ fontSize:10, fontWeight:600, color:"var(--w2)", textTransform:"uppercase", letterSpacing:".6px", marginBottom:8 }}>GPU Log</div>
            <div ref={logRef} style={{ background:"var(--k)", border:"1px solid var(--b1)", borderRadius:12,
              padding:14, height:240, overflowY:"auto", fontFamily:"var(--fm)", fontSize:11, lineHeight:1.65 }}>
              {logs.map((line, i) => (
                <div key={i} style={{ padding:"3px 0", borderBottom:"1px solid var(--b1)", color:"var(--w2)" }}
                  dangerouslySetInnerHTML={{ __html: colorLog(line) }} />
              ))}
            </div>
          </div>
        )}

        {/* ── STEP 6: Deploy ── */}
        {step === 6 && (
          <div>
            <div style={{ background:"linear-gradient(135deg,rgba(212,168,67,.07),rgba(38,201,176,.05))",
              border:"1px solid rgba(212,168,67,.18)", borderRadius:16, padding:36, textAlign:"center",
              marginBottom:20, position:"relative", overflow:"hidden" }}>
              <div style={{ position:"absolute", fontSize:120, opacity:.04, right:-20, top:-20, transform:"rotate(15deg)" }}>🦋</div>
              <div style={{ fontSize:52, marginBottom:14 }}>🦋</div>
              <div style={{ fontFamily:"var(--fh)", fontSize:30, fontWeight:700, marginBottom:8, letterSpacing:"-.3px" }}>Model Trained!</div>
              <div style={{ color:"var(--w2)", fontSize:14, lineHeight:1.6 }}>
                <strong>{spec?.model_name}</strong> achieved{" "}
                <strong style={{ color:"var(--t)" }}>{fmt(metrics.bestAcc*100)}% val accuracy</strong>
                {selDS && <> on <strong>{selDS.name}</strong></>}
                {valResult?.wasFixed && (
                  <><br /><span style={{ fontSize:12, color:"var(--g)" }}>✦ {valResult.changes?.length} consistency issues auto-fixed before training</span></>
                )}
              </div>
            </div>

            {jobs[0]?.driveLink && (
              <div style={{ background:"rgba(66,133,244,.06)", border:"1px solid rgba(66,133,244,.18)",
                borderRadius:12, padding:"13px 16px", marginBottom:20, display:"flex", alignItems:"center", gap:11 }}>
                <span style={{ fontSize:18 }}>☁️</span>
                <div><div style={{ fontSize:13, fontWeight:600, marginBottom:2 }}>Saved to Google Drive</div>
                <div style={{ fontSize:11, color:"var(--w2)" }}>best_model.pth + streamlit_app.py</div></div>
                <a href={jobs[0].driveLink} target="_blank" rel="noreferrer"
                  style={{ marginLeft:"auto", color:"#4285f4", fontSize:12, textDecoration:"none", fontWeight:600 }}>Open →</a>
              </div>
            )}

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:13, marginBottom:20 }}>
              {[
                { i:"📦", t:"Download Files", d:"train.py + config.json + classes.txt",
                  fn: () => { dl(`# ButterflAI ${jobId}\n\n${trainPy}`, `${jobId}_train.py`); dl(cfgJson, `${jobId}_config.json`) } },
                { i:"🗂", t:"Full Job Bundle", d:"All generated files",
                  fn: () => dl(`${trainPy}\n\n// config.json\n${cfgJson}`, `${jobId}_bundle.txt`) },
                { i:"📊", t:"Training History", d:"Job history + epoch metrics",
                  fn: () => setHistOpen(true) },
                { i:"🔄", t:"New Training Job", d:"Start from scratch",
                  fn: () => { setStep(1); setPlan(null); setSelDS(null); setSpec(null); setValResult(null); setTrainPy(""); setLogs([]); setProfile(null) } },
              ].map(({ i,t,d,fn }) => (
                <div key={t} onClick={fn} style={{ background:"var(--k3)", border:"1.5px solid var(--b1)",
                  borderRadius:16, padding:20, cursor:"pointer", transition:"all .25s" }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor="rgba(212,168,67,.28)"; e.currentTarget.style.transform="translateY(-2px)" }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor="var(--b1)"; e.currentTarget.style.transform="translateY(0)" }}>
                  <div style={{ fontSize:28, marginBottom:10 }}>{i}</div>
                  <div style={{ fontSize:14, fontWeight:600, marginBottom:4 }}>{t}</div>
                  <div style={{ fontSize:12, color:"var(--w2)", lineHeight:1.5 }}>{d}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Orbs() {
  return (
    <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:0, overflow:"hidden" }}>
      {[{ s:700, c:"#d4a843", t:"-350px", r:"-200px", a:"ob1 30s", o:.055 },
        { s:600, c:"#9b7ff4", b:"-300px", l:"-150px", a:"ob2 36s", o:.055 },
        { s:450, c:"#26c9b0", t:"30%", r:"3%",       a:"ob3 24s", o:.045 }].map((o,i) => (
        <div key={i} style={{ position:"absolute", width:o.s, height:o.s, borderRadius:"50%",
          background:o.c, filter:"blur(110px)", opacity:o.o,
          top:o.t, right:o.r, bottom:o.b, left:o.l,
          animation:`${o.a} ease-in-out infinite` }} />
      ))}
    </div>
  )
}

function BfMark() {
  return (
    <svg width="34" height="22" viewBox="0 0 40 26" fill="none">
      <path d="M20 13 C13 4 0 0 0 9 C0 18 10 22 20 13Z" fill="#d4a843" opacity=".95"/>
      <path d="M20 13 C27 4 40 0 40 9 C40 18 30 22 20 13Z" fill="#26c9b0" opacity=".95"/>
      <path d="M20 13 C13 20 1 23 2 17 C3 11 12 11 20 13Z" fill="#d4a843" opacity=".5"/>
      <path d="M20 13 C27 20 39 23 38 17 C37 11 28 11 20 13Z" fill="#26c9b0" opacity=".5"/>
      <ellipse cx="20" cy="13" rx="2.2" ry="2.8" fill="#eee9e0" opacity=".9"/>
    </svg>
  )
}

function AiChip({ ai }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:5, padding:"4px 12px", borderRadius:99,
      background:"var(--k3)", border:"1px solid var(--b2)", fontSize:11, color:"var(--w2)" }}>
      <span style={{ color:ai.color, fontWeight:700 }}>{ai.icon}</span>
      {ai.name}
      <span style={{ fontSize:9, padding:"1px 5px", borderRadius:99, background:ai.bg, color:ai.color, fontWeight:700 }}>{ai.badge}</span>
    </div>
  )
}

function DatasetRow({ ds, selected, onClick }) {
  const pct = ds.relevance_pct || 75
  const C   = 2 * Math.PI * 16
  const col = pct > 85 ? "var(--t)" : pct > 70 ? "var(--g)" : "var(--r)"
  return (
    <div onClick={onClick} style={{ display:"flex", alignItems:"flex-start", gap:13,
      background:"var(--k3)", border:`1.5px solid ${selected ? "rgba(212,168,67,.4)" : "var(--b1)"}`,
      borderRadius:12, padding:"14px 15px", cursor:"pointer", marginBottom:9,
      ...(selected ? { background:"rgba(212,168,67,.03)" } : {}) }}>
      <div style={{ width:15, height:15, borderRadius:"50%", border:`2px solid ${selected ? "var(--g)" : "var(--b2)"}`,
        background: selected ? "var(--g)" : "transparent", display:"flex", alignItems:"center",
        justifyContent:"center", marginTop:3, flexShrink:0 }}>
        {selected && <div style={{ width:5, height:5, borderRadius:"50%", background:"var(--k)" }} />}
      </div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontWeight:600, fontSize:13, marginBottom:5, display:"flex", alignItems:"center", gap:7, flexWrap:"wrap" }}>
          {ds.name}
          <span style={{ fontSize:9, padding:"2px 7px", borderRadius:4, fontWeight:700,
            background: ds.source === "kaggle" ? "rgba(52,211,153,.1)" : "rgba(251,191,36,.1)",
            color: ds.source === "kaggle" ? "#34d399" : "#fbbf24",
            border: ds.source === "kaggle" ? "1px solid rgba(52,211,153,.2)" : "1px solid rgba(251,191,36,.2)" }}>
            {ds.source === "kaggle" ? "KAGGLE" : "HF"}
          </span>
          {ds.size_gb > 0 && <span style={{ fontSize:9, padding:"2px 7px", borderRadius:4, background:"var(--k4)", color:"var(--w2)", border:"1px solid var(--b1)" }}>{ds.size_gb}GB</span>}
          {ds.download_count > 0 && <span style={{ fontSize:9, color:"var(--w3)" }}>↓ {ds.download_count.toLocaleString()}</span>}
        </div>
        <div style={{ display:"flex", gap:10, flexWrap:"wrap", marginBottom:4 }}>
          {ds.num_rows > 0 && <span style={{ fontSize:11, color:"var(--w2)" }}>{ds.num_rows.toLocaleString()} rows</span>}
          <span style={{ fontSize:11, color:"var(--w2)" }}>{ds.license}</span>
          {ds.tags?.slice(0,3).map(t => <span key={t} style={{ fontSize:11, color:"var(--w3)" }}>{t}</span>)}
        </div>
        <div style={{ fontSize:11, color:"var(--w3)", lineHeight:1.4 }}>{ds.description?.slice(0,180)}</div>
        {ds.download_cmd && (
          <div style={{ fontFamily:"var(--fm)", fontSize:9, color:"var(--w3)", background:"var(--k)",
            border:"1px solid var(--b1)", padding:"2px 7px", borderRadius:4, display:"inline-block", marginTop:5 }}>
            {ds.download_cmd}
          </div>
        )}
        {ds.warnings?.length > 0 && (
          <div style={{ fontSize:10, color:"var(--r)", marginTop:4 }}>⚠ {ds.warnings.join(" · ")}</div>
        )}
      </div>
      <div style={{ position:"relative", width:44, height:44, flexShrink:0 }}>
        <svg width="44" height="44" viewBox="0 0 44 44" style={{ transform:"rotate(-90deg)" }}>
          <circle cx="22" cy="22" r="16" fill="none" stroke="rgba(255,255,255,.05)" strokeWidth="3"/>
          <circle cx="22" cy="22" r="16" fill="none" stroke={col} strokeWidth="3"
            strokeDasharray={C.toFixed(1)} strokeDashoffset={(C*(1-pct/100)).toFixed(1)} strokeLinecap="round"/>
        </svg>
        <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center",
          justifyContent:"center", fontSize:11, fontWeight:700, color:col }}>{pct}%</div>
      </div>
    </div>
  )
}

function ConfigPanel({ spec, onSpec }) {
  const u = (k, v) => onSpec(p => ({ ...p, [k]: v }))
  const sel = { ...S.inp, appearance:"none", cursor:"pointer",
    backgroundImage:"url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2358534e' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E\")",
    backgroundRepeat:"no-repeat", backgroundPosition:"right 10px center", backgroundSize:"14px", paddingRight:34 }
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
      <div style={{ gridColumn:"1/-1" }}>
        <div style={{ fontSize:11, color:"var(--w2)", marginBottom:5 }}>Model Architecture</div>
        <select style={sel} value={spec.model_name} onChange={e => u("model_name", e.target.value)}>
          {["resnet18","resnet50","resnet101","efficientnet_b0","efficientnet_b3",
            "mobilenet_v3_large","vit_b_16","convnext_small"].map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>
      {[["Epochs","epochs",1,100,1],["Batch Size","batch_size",4,128,4]].map(([l,k,mn,mx,st]) => (
        <div key={k}>
          <div style={{ fontSize:11, color:"var(--w2)", marginBottom:5 }}>{l}</div>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <input type="range" min={mn} max={mx} step={st} value={spec[k]}
              onChange={e => u(k, parseInt(e.target.value))} style={{ flex:1, accentColor:"var(--g)" }} />
            <span style={{ minWidth:36, fontFamily:"var(--fm)", fontSize:12, color:"var(--g)", textAlign:"right" }}>{spec[k]}</span>
          </div>
        </div>
      ))}
      {[["Learning Rate","lr",{"0.001":"1e-3","0.0001":"1e-4 ✓","0.00003":"3e-5","0.00001":"1e-5"}],
        ["Optimizer","optimizer",{"adamw":"AdamW ✓","sgd":"SGD","adam":"Adam"}],
        ["Scheduler","scheduler",{"cosine":"Cosine ✓","onecycle":"OneCycle","step":"StepLR","none":"None"}],
        ["Augmentation","augmentation",{"light":"Light","standard":"Standard ✓","heavy":"Heavy","mixup":"MixUp"}],
        ["Image Size","image_size",{"128":"128×128","224":"224×224 ✓","299":"299×299","384":"384×384"}],
        ["Mixed Precision","fp16",{"true":"FP16 ✓","false":"FP32"}],
        ["Early Stopping","early_stopping_patience",{"5":"Patience 5 ✓","10":"Patience 10","0":"Off"}],
        ["Pretrained","pretrained",{"imagenet":"ImageNet ✓","none":"Scratch"}],
      ].map(([label, key, opts]) => (
        <div key={key}>
          <div style={{ fontSize:11, color:"var(--w2)", marginBottom:5 }}>{label}</div>
          <select style={sel} value={String(spec[key])}
            onChange={e => { const v = e.target.value; u(key, v==="true"?true:v==="false"?false:isNaN(v)?v:Number(v)) }}>
            {Object.entries(opts).map(([v,l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </div>
      ))}
    </div>
  )
}

function ConsistencyBanner({ valResult }) {
  if (!valResult?.wasFixed || !valResult?.changes?.length) return null
  return (
    <div style={{ background:"linear-gradient(135deg,rgba(212,168,67,.07),rgba(38,201,176,.04))",
      border:"1px solid rgba(212,168,67,.22)", borderRadius:14, padding:20, marginBottom:20 }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
        <div style={{ fontFamily:"var(--fh)", fontSize:18, fontWeight:600, color:"var(--g)" }}>
          Auto-Fixed {valResult.changes.length} Issue{valResult.changes.length > 1 ? "s" : ""}
        </div>
        <span style={{ fontSize:9, padding:"2px 8px", borderRadius:99, background:"rgba(212,168,67,.15)",
          color:"var(--g)", border:"1px solid rgba(212,168,67,.25)", fontWeight:700, letterSpacing:".4px" }}>
          AUTO-REPAIRED
        </span>
      </div>
      {valResult.changes.map((c, i) => (
        <div key={i} style={{ background:"var(--k3)", border:"1px solid rgba(212,168,67,.15)",
          borderRadius:10, padding:"12px 14px", marginBottom:8 }}>
          <div style={{ fontSize:12, fontWeight:600, color:"var(--g)", marginBottom:4 }}>✦ {c.title}</div>
          <div style={{ fontSize:11, color:"var(--w2)", lineHeight:1.6, marginBottom: c.before||c.after ? 8 : 0 }}>{c.reason}</div>
          {(c.before || c.after) && (
            <div style={{ display:"flex", alignItems:"center", gap:7, flexWrap:"wrap" }}>
              {c.before && <span style={{ fontFamily:"var(--fm)", fontSize:10, padding:"2px 8px", borderRadius:4,
                background:"rgba(232,96,122,.08)", color:"var(--r)", border:"1px solid rgba(232,96,122,.2)", textDecoration:"line-through" }}>{c.before}</span>}
              {c.before && c.after && <span style={{ color:"var(--w3)", fontSize:12 }}>→</span>}
              {c.after && <span style={{ fontFamily:"var(--fm)", fontSize:10, padding:"2px 8px", borderRadius:4,
                background:"rgba(38,201,176,.08)", color:"var(--t)", border:"1px solid rgba(38,201,176,.2)" }}>{c.after}</span>}
            </div>
          )}
        </div>
      ))}
      {valResult.diff?.length > 0 && <DiffView diff={valResult.diff} />}
    </div>
  )
}

function CheckGrid({ checks }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:8, marginBottom:20 }}>
      {checks.map(c => (
        <div key={c.id} style={{ display:"flex", alignItems:"flex-start", gap:11,
          background:"var(--k3)", borderRadius:11, padding:"11px 14px",
          border:`1px solid ${c.status==="pass" ? "rgba(38,201,176,.12)" : c.status==="warn" ? "rgba(251,191,36,.12)" : "rgba(232,96,122,.15)"}` }}>
          <div style={{ width:25, height:25, borderRadius:"50%", display:"flex", alignItems:"center",
            justifyContent:"center", fontSize:11, fontWeight:700, flexShrink:0,
            background: c.status==="pass" ? "rgba(38,201,176,.1)" : c.status==="warn" ? "rgba(251,191,36,.1)" : "rgba(232,96,122,.1)",
            border: `1px solid ${c.status==="pass" ? "rgba(38,201,176,.25)" : c.status==="warn" ? "rgba(251,191,36,.25)" : "rgba(232,96,122,.25)"}`,
            color: c.status==="pass" ? "var(--t)" : c.status==="warn" ? "#fbbf24" : "var(--r)" }}>
            {c.status==="pass" ? "✓" : c.status==="warn" ? "⚠" : "✕"}
          </div>
          <div>
            <div style={{ fontSize:13, fontWeight:600, marginBottom:3 }}>{c.title}</div>
            <div style={{ fontSize:12, color:"var(--w2)", lineHeight:1.5 }}>{c.detail}</div>
            {c.fix_description && c.status !== "pass" && (
              <div style={{ fontSize:11, color:"var(--g)", marginTop:4, fontStyle:"italic" }}>↳ Fixed: {c.fix_description}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function DiffView({ diff }) {
  const [open, setOpen] = useState(true)
  return (
    <div style={{ marginTop:14 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:8 }}>
        <span style={{ fontSize:11, fontWeight:600, color:"var(--w2)", textTransform:"uppercase", letterSpacing:".5px" }}>Code Diff</span>
        <button onClick={() => setOpen(o => !o)} style={{ background:"none", border:"none", color:"var(--w3)", fontSize:11, cursor:"pointer", fontFamily:"var(--fb)" }}>{open ? "Hide" : "Show"}</button>
      </div>
      {open && (
        <div style={{ background:"var(--k)", border:"1px solid var(--b1)", borderRadius:8, fontFamily:"var(--fm)",
          fontSize:11, overflow:"auto", maxHeight:220, lineHeight:1.7 }}>
          {diff.map((l, i) => (
            <div key={i} style={{ padding:"1px 14px", display:"flex", gap:10,
              borderLeft:`3px solid ${l.type==="added"?"var(--t)":l.type==="removed"?"var(--r)":"transparent"}`,
              background:l.type==="added"?"rgba(38,201,176,.06)":l.type==="removed"?"rgba(232,96,122,.06)":"transparent",
              color:l.type==="added"?"var(--t)":l.type==="removed"?"var(--r)":l.type==="ellipsis"?"var(--w3)":"var(--w3)",
              textDecoration:l.type==="removed"?"line-through":"none", fontStyle:l.type==="ellipsis"?"italic":"normal",
              opacity:l.type==="removed"?0.7:1 }}>
              {l.type !== "ellipsis" && <span style={{ color:"var(--w3)", minWidth:28, textAlign:"right" }}>{l.lineNum}</span>}
              <span>{l.line}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function CodeTabs({ trainPy, cfgJson, classes, tab, onTab }) {
  const tabs = [["train.py", trainPy], ["config.json", cfgJson], ["classes.txt", classes]]
  return (
    <>
      <div style={{ fontFamily:"var(--fh)", fontSize:17, fontWeight:600, marginBottom:12, letterSpacing:"-.2px" }}>Generated Code</div>
      <div style={{ display:"flex", gap:2, background:"var(--k3)", padding:4, borderRadius:8, marginBottom:10 }}>
        {tabs.map(([t], i) => (
          <button key={t} onClick={() => onTab(i)} style={{ flex:1, padding:7, textAlign:"center",
            fontSize:11, fontWeight:500, borderRadius:6, cursor:"pointer",
            color: tab===i ? "var(--w1)" : "var(--w2)",
            background: tab===i ? "var(--k2)" : "none", border:"none", fontFamily:"var(--fb)" }}>{t}</button>
        ))}
      </div>
      <pre style={{ background:"var(--k)", border:"1px solid var(--b1)", borderRadius:8, padding:14,
        fontFamily:"var(--fm)", fontSize:11, color:"var(--w2)", overflow:"auto", maxHeight:320,
        whiteSpace:"pre", lineHeight:1.65 }}>{tabs[tab][1] || "Generating…"}</pre>
    </>
  )
}

function ProfileSummary({ profile }) {
  if (!profile) return null
  const fa = profile.format_analysis || {}
  const ca = profile.class_analysis  || {}
  const da = profile.domain_analysis || {}
  const w  = profile.warnings        || []
  return (
    <div style={{ background:"rgba(38,201,176,.04)", border:"1px solid rgba(38,201,176,.15)",
      borderRadius:12, padding:"14px 16px", marginTop:16, marginBottom:4 }}>
      <div style={{ fontSize:12, fontWeight:600, color:"var(--t)", marginBottom:10 }}>📊 Dataset Intelligence Report</div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10, marginBottom:10 }}>
        {[
          ["Modality", profile.modality],
          ["Domain",   da.type],
          ["Total files", (profile.statistics?.total_files || 0).toLocaleString()],
          ["Imbalance", `${ca.imbalance_ratio || 1}x`],
          ["Grayscale",  fa.is_grayscale ? "Yes" : "No"],
          ["DICOM",     fa.has_dicom ? "Yes" : "No"],
        ].map(([l,v]) => v != null && (
          <div key={l}>
            <div style={{ fontSize:9, textTransform:"uppercase", letterSpacing:1, color:"var(--w3)", marginBottom:3 }}>{l}</div>
            <div style={{ fontSize:12, fontWeight:500 }}>{String(v)}</div>
          </div>
        ))}
      </div>
      {w.length > 0 && w.map((warning, i) => (
        <div key={i} style={{ fontSize:11, color:"var(--r)", marginTop:4 }}>⚠ {warning}</div>
      ))}
    </div>
  )
}

function LiveCharts({ chart }) {
  const base = {
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{ labels:{ color:"rgba(255,255,255,.3)", boxWidth:8, font:{size:10} } } },
    scales:{ x:{ grid:{color:"rgba(255,255,255,.03)"}, ticks:{color:"rgba(255,255,255,.25)",font:{size:9},maxTicksLimit:7} },
             y:{ grid:{color:"rgba(255,255,255,.03)"}, ticks:{color:"rgba(255,255,255,.25)",font:{size:9}} } },
    animation:{duration:0},
  }
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:20 }}>
      <div style={{ background:"var(--k3)", border:"1px solid var(--b1)", borderRadius:12, padding:15 }}>
        <div style={{ fontSize:10, fontWeight:600, color:"var(--w2)", textTransform:"uppercase", letterSpacing:".5px", marginBottom:10 }}>Accuracy (%)</div>
        <div style={{ height:160 }}>
          <Line options={{ ...base, scales:{ ...base.scales, y:{...base.scales.y,min:0,max:100} } }}
            data={{ labels:chart.labels, datasets:[
              { label:"Train", data:chart.acc,  borderColor:"#d4a843", backgroundColor:"rgba(212,168,67,.06)", borderWidth:1.5, pointRadius:0, tension:.4, fill:true },
              { label:"Val",   data:chart.vacc, borderColor:"#26c9b0", backgroundColor:"rgba(38,201,176,.06)", borderWidth:1.5, pointRadius:0, tension:.4, fill:true },
            ] }} />
        </div>
      </div>
      <div style={{ background:"var(--k3)", border:"1px solid var(--b1)", borderRadius:12, padding:15 }}>
        <div style={{ fontSize:10, fontWeight:600, color:"var(--w2)", textTransform:"uppercase", letterSpacing:".5px", marginBottom:10 }}>Loss</div>
        <div style={{ height:160 }}>
          <Line options={base}
            data={{ labels:chart.labels, datasets:[
              { label:"Train", data:chart.loss,  borderColor:"#e8607a", backgroundColor:"rgba(232,96,122,.06)", borderWidth:1.5, pointRadius:0, tension:.4, fill:true },
              { label:"Val",   data:chart.vloss, borderColor:"rgba(232,96,122,.4)", borderWidth:1, pointRadius:0, tension:.4, borderDash:[4,4] },
            ] }} />
        </div>
      </div>
    </div>
  )
}

function colorLog(line) {
  return (line || "")
    .replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/(\d+\.\d+%)/g,            '<span style="color:var(--t)">$1</span>')
    .replace(/(val_loss=[\d.]+|train_loss=[\d.]+)/g, '<span style="color:var(--r)">$1</span>')
    .replace(/(\[EPOCH:\d+\/\d+\])/g,   '<span style="color:var(--g);font-weight:600">$1</span>')
    .replace(/(★ BEST)/g,               '<span style="color:var(--g)">$1</span>')
    .replace(/(\[ButterflAI\])/g,        '<span style="color:var(--w3)">$1</span>')
    .replace(/(\[DataIntel\])/g,         '<span style="color:var(--t)">$1</span>')
    .replace(/(ERROR:.*)/g,             '<span style="color:var(--r)">$1</span>')
    .replace(/(WARNING:|⚠)/g,           '<span style="color:#fbbf24">$1</span>')
}

// ── Sign-in page ───────────────────────────────────────────────────────────────
function SignIn({ onGoogle, onManual, ai, onAi }) {
  const [form, setForm] = useState({
    kaggleUser:"", kaggleKey:"", hfToken:"", modalToken:"", modalSecret:"", driveFolder:"", apiKey:"",
  })
  const [showPw, setShowPw] = useState({})
  const [err, setErr]       = useState("")
  const tp = id => setShowPw(p => ({...p, [id]: !p[id]}))
  const up = (k,v) => setForm(p => ({...p, [k]:v}))

  const submit = () => {
    if (!form.kaggleUser.trim() && !form.apiKey.trim()) { setErr("Kaggle username OR an AI API key is required."); return }
    setErr("")
    onManual({ ...form, ai })
  }

  const selStyle = { background:"var(--k3)", border:"1.5px solid var(--b1)", borderRadius:12,
    cursor:"pointer", textAlign:"center", padding:"10px 8px", transition:"all .2s" }
  const selActiveStyle = { borderColor:"rgba(212,168,67,.4)", background:"rgba(212,168,67,.06)" }

  return (
    <div style={{ ...S.page }}>
      <Orbs />
      <div style={{ ...S.wrap }}>
        <header style={S.hdr}>
          <div style={S.logo}><BfMark /><div style={S.logoT}><span style={{color:"var(--g)"}}>Butterfl</span><span style={{color:"var(--t)",fontStyle:"italic"}}>AI</span></div></div>
        </header>
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", minHeight:"78vh", paddingTop:56, textAlign:"center" }}>
          <div style={{ fontSize:11, letterSpacing:3, textTransform:"uppercase", color:"var(--g)", fontWeight:600, marginBottom:18 }}>↯ End-to-End ML Training Platform</div>
          <h1 style={{ fontFamily:"var(--fh)", fontSize:"clamp(42px,7.5vw,78px)", fontWeight:700, lineHeight:1.05, letterSpacing:"-2px", marginBottom:8 }}>
            Train any model.<br /><em style={{color:"var(--g)",fontStyle:"italic"}}>No DevOps. No PhD.</em>
          </h1>
          <p style={{ color:"var(--w2)", fontSize:15, fontWeight:300, lineHeight:1.75, maxWidth:520, margin:"16px auto 36px" }}>
            Type one sentence. ButterflAI searches <strong style={{color:"var(--w1)"}}>real Kaggle + HuggingFace datasets</strong>,
            generates production code, auto-fixes consistency issues, trains on real T4 GPUs, and delivers a working model with a Streamlit demo.
          </p>

          <div style={{ width:"100%", maxWidth:420 }}>
            <button style={{ display:"flex", alignItems:"center", gap:12, width:"100%", padding:"14px 20px",
              background:"rgba(255,255,255,.04)", border:"1.5px solid var(--b2)", borderRadius:12,
              color:"var(--w1)", fontFamily:"var(--fb)", fontSize:14, fontWeight:500, cursor:"pointer", marginBottom:14 }}
              onClick={onGoogle}>
              <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
              Continue with Google
              <span style={{ marginLeft:"auto", fontSize:11, color:"var(--w3)" }}>Drive + auth</span>
            </button>

            <div style={{ display:"flex", alignItems:"center", gap:10, margin:"12px 0", color:"var(--w3)", fontSize:11 }}>
              <div style={{ flex:1, height:1, background:"var(--b1)" }} /><span>or API keys</span><div style={{ flex:1, height:1, background:"var(--b1)" }} />
            </div>

            <div style={{ fontSize:12, color:"var(--w2)", marginBottom:9 }}>AI Provider — <span style={{color:"var(--t)",fontWeight:600}}>all free tiers</span></div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, marginBottom:14 }}>
              {Object.entries(AI).map(([k,a]) => (
                <div key={k} onClick={() => onAi(k)} style={{ ...selStyle, ...(ai===k ? selActiveStyle : {}) }}>
                  <div style={{ fontSize:20, marginBottom:4, color:a.color, fontWeight:700 }}>{a.icon}</div>
                  <div style={{ fontSize:11, fontWeight:600, marginBottom:3 }}>{a.name.split(" ")[0]}</div>
                  <span style={{ fontSize:9, padding:"1px 6px", borderRadius:99, background:a.bg, color:a.color, fontWeight:700 }}>{a.badge}</span>
                </div>
              ))}
            </div>

            <Field label={<>{AI[ai].name} API Key <a href={ai==="gemini"?"https://aistudio.google.com/app/apikey":ai==="groq"?"https://console.groq.com":"https://console.anthropic.com"} target="_blank" rel="noreferrer" style={{color:"var(--t)",fontSize:10,textDecoration:"none"}}>Get free →</a></>}>
              <PwInp id="ak" val={form.apiKey} onChange={v=>up("apiKey",v)} show={showPw.ak} onToggle={()=>tp("ak")}
                placeholder={ai==="gemini"?"AIzaxxxxxxxx":ai==="groq"?"gsk_xxxxxxxx":"sk-ant-xxxxxxxx"} />
            </Field>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:11 }}>
              <Field label={<>Kaggle Username <a href="https://www.kaggle.com/settings/account" target="_blank" rel="noreferrer" style={{color:"var(--w3)",fontSize:10,textDecoration:"none"}}>needed for real search</a></>}>
                <input style={S.inp} placeholder="your_username" value={form.kaggleUser} onChange={e=>up("kaggleUser",e.target.value)} />
              </Field>
              <Field label="Kaggle API Key">
                <PwInp id="kk" val={form.kaggleKey} onChange={v=>up("kaggleKey",v)} show={showPw.kk} onToggle={()=>tp("kk")} placeholder="xxxxxxxxx" />
              </Field>
              <Field label={<>Modal Token ID <a href="https://modal.com" target="_blank" rel="noreferrer" style={{color:"var(--a4)",fontSize:10,textDecoration:"none"}}>Free GPU →</a></>}>
                <PwInp id="mt" val={form.modalToken} onChange={v=>up("modalToken",v)} show={showPw.mt} onToggle={()=>tp("mt")} placeholder="ak-xxxxxx" />
              </Field>
              <Field label="Modal Token Secret">
                <PwInp id="ms" val={form.modalSecret} onChange={v=>up("modalSecret",v)} show={showPw.ms} onToggle={()=>tp("ms")} placeholder="xxxxxxxxx" />
              </Field>
            </div>

            {err && <div style={{ background:"rgba(232,96,122,.09)", border:"1px solid rgba(232,96,122,.2)", color:"#f87171", borderRadius:8, padding:"9px 13px", fontSize:12, marginTop:8 }}>{err}</div>}
            <button style={S.gold} onClick={submit}>Launch ButterflAI →</button>
            <div style={{ marginTop:16, paddingTop:14, borderTop:"1px solid var(--b1)", fontSize:11, color:"var(--w3)", lineHeight:1.8 }}>
              🔒 Keys stay in your browser session only — never stored server-side.<br />
              🆓 Kaggle, Gemini, Groq all have free tiers. No paid subscription required.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:5, marginBottom:12 }}>
      <label style={{ fontSize:11, color:"var(--w2)", fontWeight:500, display:"flex", alignItems:"center", justifyContent:"space-between" }}>{label}</label>
      {children}
    </div>
  )
}

function PwInp({ id, val, onChange, show, onToggle, placeholder }) {
  return (
    <div style={{ position:"relative" }}>
      <input style={S.inp} type={show?"text":"password"} value={val} onChange={e=>onChange(e.target.value)} placeholder={placeholder} />
      <button onClick={onToggle} style={{ position:"absolute", right:10, top:"50%", transform:"translateY(-50%)",
        background:"none", border:"none", cursor:"pointer", color:"var(--w3)", fontSize:14, padding:4 }}>
        {show ? "🙈" : "👁"}
      </button>
    </div>
  )
}