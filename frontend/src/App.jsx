import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws";

/* ── tiny helpers ─────────────────────────────────────────────────────────── */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt   = (n) => (typeof n === "number" ? n.toFixed(3) : "0.000");

/* ── Score Ring ──────────────────────────────────────────────────────────── */
function ScoreRing({ score, isAnomaly }) {
  const r   = 54;
  const circ = 2 * Math.PI * r;
  const pct  = clamp(score, 0, 1);
  const dash = pct * circ;
  const color = isAnomaly
    ? "#ff2d2d"
    : score > 0.25
    ? "#ffaa00"
    : "#00e5a0";

  return (
    <div style={{ position: "relative", width: 140, height: 140 }}>
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={r} fill="none" stroke="#1a1f2e" strokeWidth="12" />
        <circle
          cx="70" cy="70" r={r}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
          style={{ transition: "stroke-dasharray 0.4s ease, stroke 0.4s ease" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 26, fontWeight: 700, color, lineHeight: 1 }}>
          {fmt(score)}
        </span>
        <span style={{ fontSize: 10, color: "#4a5568", marginTop: 4, letterSpacing: 2 }}>SCORE</span>
      </div>
    </div>
  );
}

/* ── Sparkline ───────────────────────────────────────────────────────────── */
function Sparkline({ data, isAnomaly }) {
  if (!data || data.length < 2) return (
    <div style={{ height: 64, display: "flex", alignItems: "center", justifyContent: "center", color: "#2d3748" }}>
      <span style={{ fontSize: 12, letterSpacing: 2 }}>AWAITING DATA</span>
    </div>
  );

  const W = 400, H = 64;
  const max = Math.max(...data, 0.5);
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - (v / max) * (H - 8) - 4;
    return `${x},${y}`;
  }).join(" ");

  const color = isAnomaly ? "#ff2d2d" : "#00e5a0";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 64 }} preserveAspectRatio="none">
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* threshold line at 0.45 */}
      <line
        x1="0" y1={H - (0.45 / max) * (H - 8) - 4}
        x2={W} y2={H - (0.45 / max) * (H - 8) - 4}
        stroke="#ff2d2d" strokeWidth="1" strokeDasharray="4 4" opacity="0.5"
      />
      <polyline
        points={`0,${H} ${pts} ${W},${H}`}
        fill="url(#sg)" stroke="none"
      />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

/* ── Alert Row ───────────────────────────────────────────────────────────── */
function AlertRow({ alert, idx }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12,
      padding: "8px 12px",
      background: idx === 0 ? "rgba(255,45,45,0.08)" : "transparent",
      borderLeft: idx === 0 ? "2px solid #ff2d2d" : "2px solid transparent",
      transition: "all 0.3s",
    }}>
      <span style={{ color: "#ff2d2d", fontSize: 10, fontFamily: "monospace", minWidth: 60 }}>{alert.time}</span>
      <span style={{ flex: 1, color: "#e2e8f0", fontSize: 12, letterSpacing: 1 }}>{alert.label}</span>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700,
        color: alert.score > 0.7 ? "#ff2d2d" : "#ffaa00",
      }}>
        {alert.score.toFixed(3)}
      </span>
    </div>
  );
}

/* ── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  const [running,      setRunning]      = useState(false);
  const [score,        setScore]        = useState(0);
  const [label,        setLabel]        = useState("Idle");
  const [isAnomaly,    setIsAnomaly]    = useState(false);
  const [framesProc,   setFramesProc]   = useState(0);
  const [scoreHistory, setScoreHistory] = useState([]);
  const [alerts,       setAlerts]       = useState([]);
  const [device,       setDevice]       = useState("N/A");
  const [connStatus,   setConnStatus]   = useState("disconnected"); // connected | disconnected | error

  // Config
  const [source,      setSource]      = useState("http://100.80.253.25:8080/video");
  const [bufferSize,  setBufferSize]  = useState(32);
  const [skip,        setSkip]        = useState(3);
  const [threshold,   setThreshold]   = useState(0.45);

  const wsRef    = useRef(null);
  const pingRef  = useRef(null);

  /* WebSocket */
  const connectWS = useCallback(() => {
  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
    return;
  }

  const ws = new WebSocket(WS);

  ws.onopen = () => setConnStatus("connected");

  ws.onclose = () => {
    setConnStatus("disconnected");
    setTimeout(() => connectWS(), 2000);
  };

  ws.onerror = () => setConnStatus("error");

  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      setScore(d.score ?? 0);
      setLabel(d.label ?? "");
      setIsAnomaly(d.is_anomaly ?? false);
      setFramesProc(d.frames_proc ?? 0);
      setScoreHistory(d.score_history ?? []);
      setAlerts(d.alerts ?? []);
      setDevice(d.device ?? "N/A");
      setRunning(d.running ?? false);
    } catch {}
  };

  wsRef.current = ws;
}, []);

  useEffect(() => {
    connectWS();
    return () => wsRef.current?.close();
  }, [connectWS]);

  /* Keep-alive ping */
  useEffect(() => {
    pingRef.current = setInterval(() => {
      if (wsRef.current?.readyState === 1) wsRef.current.send("ping");
    }, 5000);
    return () => clearInterval(pingRef.current);
  }, []);

  const handleStart = async () => {
    await fetch(`${API}/api/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, buffer_size: bufferSize, skip, threshold }),
    });
    setRunning(true);
  };

  const handleStop = async () => {
    await fetch(`${API}/api/stop`, { method: "POST" });
    setRunning(false);
  };

  const clearAlerts = async () => {
    await fetch(`${API}/api/clear_alerts`, { method: "POST" });
    setAlerts([]);
  };

  const statusColor = { connected: "#00e5a0", disconnected: "#4a5568", error: "#ff2d2d" }[connStatus];

  return (
    <>
      {/* Google Font import */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #090c14; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }

        input[type=text], input[type=number] {
          background: #111827;
          border: 1px solid #1e2533;
          color: #e2e8f0;
          padding: 8px 12px;
          border-radius: 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          width: 100%;
          outline: none;
          transition: border 0.2s;
        }
        input[type=text]:focus, input[type=number]:focus { border-color: #00e5a0; }

        .btn {
          padding: 10px 24px;
          border: none;
          border-radius: 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 2px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-start {
          background: #00e5a0;
          color: #090c14;
        }
        .btn-start:hover { background: #00ffb3; transform: translateY(-1px); }
        .btn-stop {
          background: transparent;
          color: #ff2d2d;
          border: 1px solid #ff2d2d;
        }
        .btn-stop:hover { background: rgba(255,45,45,0.1); }
        .btn-ghost {
          background: transparent;
          color: #4a5568;
          border: 1px solid #1e2533;
          padding: 6px 14px;
          font-size: 11px;
        }
        .btn-ghost:hover { border-color: #4a5568; color: #a0aec0; }

        .card {
          background: #0d1117;
          border: 1px solid #1e2533;
          border-radius: 10px;
          padding: 20px;
        }

        .label { font-size: 10px; color: #4a5568; letter-spacing: 3px; margin-bottom: 8px; }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111827; }
        ::-webkit-scrollbar-thumb { background: #1e2533; border-radius: 4px; }
      `}</style>

      {/* ── Top Bar ── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 28px",
        borderBottom: "1px solid #1e2533",
        background: "#090c14",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, #00e5a0, #0066ff)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>⬡</div>
          <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 800, letterSpacing: 1 }}>
            DSA<span style={{ color: "#00e5a0" }}>NET</span>
          </span>
          <span style={{ color: "#2d3748", fontSize: 12 }}>// Anomaly Detection</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: statusColor, boxShadow: `0 0 6px ${statusColor}` }} />
            <span style={{ fontSize: 11, color: statusColor, letterSpacing: 2 }}>{connStatus.toUpperCase()}</span>
          </div>
          <div style={{ fontSize: 11, color: "#2d3748" }}>
            DEVICE: <span style={{ color: "#4a5568" }}>{device}</span>
          </div>
          <div style={{ fontSize: 11, color: "#2d3748" }}>
            FRAMES: <span style={{ color: "#4a5568" }}>{framesProc.toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* ── Main Layout ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16, padding: "16px 24px", height: "calc(100vh - 61px)" }}>

        {/* ── LEFT COLUMN ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14, overflow: "hidden" }}>

          {/* Video Feed */}
          <div className="card" style={{ flex: "1 1 auto", overflow: "hidden", position: "relative", minHeight: 0 }}>
            <div className="label">LIVE FEED</div>

            {running ? (
              <img
                src={`${API}/video_feed`}
                alt="Live feed"
                style={{ width: "100%", height: "calc(100% - 24px)", objectFit: "contain", borderRadius: 6, background: "#000" }}
              />
            ) : (
              <div style={{
                height: "calc(100% - 24px)", display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                background: "#060810", borderRadius: 6,
                border: "1px dashed #1e2533",
              }}>
                <div style={{ fontSize: 40, marginBottom: 16, opacity: 0.2 }}>⬡</div>
                <div style={{ color: "#2d3748", fontSize: 12, letterSpacing: 3 }}>FEED OFFLINE</div>
                <div style={{ color: "#1e2533", fontSize: 10, marginTop: 8 }}>Configure source and press START</div>
              </div>
            )}

            {/* Anomaly Overlay */}
            {isAnomaly && running && (
              <div style={{
                position: "absolute", inset: 0,
                border: "3px solid #ff2d2d",
                borderRadius: 10,
                pointerEvents: "none",
                animation: "pulse 1s ease-in-out infinite",
              }} />
            )}
            <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
          </div>

          {/* Sparkline */}
          <div className="card" style={{ flex: "0 0 auto" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div className="label" style={{ margin: 0 }}>ANOMALY SCORE HISTORY</div>
              <div style={{ fontSize: 10, color: "#2d3748" }}>
                <span style={{ color: "#ff2d2d" }}>— </span>threshold {threshold}
              </div>
            </div>
            <Sparkline data={scoreHistory} isAnomaly={isAnomaly} />
          </div>
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14, overflowY: "auto" }}>

          {/* Score + Status */}
          <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
            <ScoreRing score={score} isAnomaly={isAnomaly} />

            <div style={{
              padding: "10px 24px",
              borderRadius: 6,
              background: isAnomaly ? "rgba(255,45,45,0.12)" : "rgba(0,229,160,0.08)",
              border: `1px solid ${isAnomaly ? "#ff2d2d" : "#00e5a0"}`,
              textAlign: "center",
              transition: "all 0.3s",
              width: "100%",
            }}>
              <div style={{
                fontFamily: "'Syne', sans-serif",
                fontWeight: 800,
                fontSize: 16,
                color: isAnomaly ? "#ff2d2d" : "#00e5a0",
                letterSpacing: 2,
              }}>
                {isAnomaly ? "⚠ CRIME DETECTED" : label}
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="card">
            <div className="label">CONFIGURATION</div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ fontSize: 10, color: "#4a5568", marginBottom: 4 }}>VIDEO SOURCE</div>
                <input
                  type="text"
                  value={source}
                  onChange={e => setSource(e.target.value)}
                  placeholder="0 | rtsp:// | http:// | file.mp4"
                  disabled={running}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568", marginBottom: 4 }}>BUFFER</div>
                  <input type="number" value={bufferSize} onChange={e => setBufferSize(+e.target.value)} min={8} max={128} disabled={running} />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568", marginBottom: 4 }}>SKIP</div>
                  <input type="number" value={skip} onChange={e => setSkip(+e.target.value)} min={1} max={10} disabled={running} />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568", marginBottom: 4 }}>THRESHOLD</div>
                  <input type="number" value={threshold} onChange={e => setThreshold(+e.target.value)} min={0.1} max={0.99} step={0.01} disabled={running} />
                </div>
              </div>

              <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                {!running ? (
                  <button className="btn btn-start" style={{ flex: 1 }} onClick={handleStart}>
                    ▶ START
                  </button>
                ) : (
                  <button className="btn btn-stop" style={{ flex: 1 }} onClick={handleStop}>
                    ■ STOP
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="card">
            <div className="label">SESSION STATS</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {[
                ["FRAMES",    framesProc.toLocaleString()],
                ["ALERTS",    alerts.length],
                ["PEAK SCORE", scoreHistory.length ? Math.max(...scoreHistory).toFixed(3) : "0.000"],
                ["AVG SCORE",  scoreHistory.length ? (scoreHistory.reduce((a,b) => a+b, 0) / scoreHistory.length).toFixed(3) : "0.000"],
              ].map(([k, v]) => (
                <div key={k} style={{ background: "#060810", borderRadius: 6, padding: "10px 12px" }}>
                  <div style={{ fontSize: 9, color: "#4a5568", letterSpacing: 2, marginBottom: 4 }}>{k}</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 18, fontWeight: 700, color: "#e2e8f0" }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Alert Log */}
          <div className="card" style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div className="label" style={{ margin: 0 }}>
                ALERT LOG
                {alerts.length > 0 && (
                  <span style={{
                    marginLeft: 8, background: "#ff2d2d", color: "#fff",
                    fontSize: 9, padding: "1px 6px", borderRadius: 10,
                  }}>{alerts.length}</span>
                )}
              </div>
              {alerts.length > 0 && (
                <button className="btn btn-ghost" onClick={clearAlerts}>CLEAR</button>
              )}
            </div>

            <div style={{ overflowY: "auto", maxHeight: 220 }}>
              {alerts.length === 0 ? (
                <div style={{ textAlign: "center", color: "#2d3748", fontSize: 11, padding: "24px 0", letterSpacing: 2 }}>
                  NO ALERTS
                </div>
              ) : (
                alerts.map((a, i) => <AlertRow key={`${a.time}-${i}`} alert={a} idx={i} />)
              )}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}