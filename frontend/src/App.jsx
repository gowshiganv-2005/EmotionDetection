import React, { useRef, useState, useEffect, useCallback } from "react";
import "./App.css";

const EMOTION_EMOJI = {
  Happy: "😄",
  Sad: "😢",
  Angry: "😠",
  Fear: "😨",
  Surprise: "😲",
  Disgust: "🤢",
  Neutral: "😐",
};

const EMOTION_COLOR = {
  Happy: "#FFD700",
  Sad: "#4FC3F7",
  Angry: "#FF5252",
  Fear: "#CE93D8",
  Surprise: "#FF9800",
  Disgust: "#A5D6A7",
  Neutral: "#B0BEC5",
};

const GENDER_EMOJI = { Man: "👨", Woman: "👩", Unknown: "🧑" };

function SongCard({ song, index }) {
  return (
    <div className="song-card" style={{ "--delay": `${index * 0.1}s` }}>
      <div className="song-rank">{index + 1}</div>
      <div className="song-info">
        <div className="song-title">{song.title}</div>
        <div className="song-meta">
          <span className="song-movie">🎬 {song.movie}</span>
          <span className="song-singer">🎤 {song.singer}</span>
        </div>
      </div>
      <a
        href={song.youtube}
        target="_blank"
        rel="noopener noreferrer"
        className="song-play-btn"
        aria-label={`Play ${song.title}`}
      >
        <span className="play-icon">▶</span>
      </a>
    </div>
  );
}

function FaceResult({ face }) {
  const emoji = EMOTION_EMOJI[face.emotion] || "😐";
  const color = EMOTION_COLOR[face.emotion] || "#B0BEC5";
  const genderEmoji = GENDER_EMOJI[face.gender] || "🧑";
  const detPct = face.det_conf ? Math.round(face.det_conf * 100) : null;

  return (
    <div className="face-result" style={{ "--emotion-color": color }}>
      <div className="face-result-header">
        <div className="emotion-badge" style={{ background: `${color}22`, border: `1.5px solid ${color}` }}>
          <span className="emotion-emoji">{emoji}</span>
          <div>
            <div className="emotion-label">{face.emotion}</div>
            <div className="emotion-conf">{face.confidence}% emotion confidence</div>
          </div>
        </div>
        <div className="info-chips">
          {detPct !== null && (
            <span className="chip yolo-chip">🎯 YOLO {detPct}%</span>
          )}
          <span className="chip gender-chip">{genderEmoji} {face.gender}</span>
          {face.age !== "N/A" && (
            <span className="chip age-chip">🎂 Age ~{face.age}</span>
          )}
        </div>
      </div>

      <div className="songs-section">
        <div className="songs-header">
          <span className="music-note">🎵</span>
          <span>Tamil Songs For You</span>
        </div>
        <div className="songs-list">
          {face.songs.map((song, i) => (
            <SongCard key={i} song={song} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [faces, setFaces] = useState([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [backendOk, setBackendOk] = useState(null);
  const [autoMode, setAutoMode] = useState(false);
  const [lastAnalyzedAt, setLastAnalyzedAt] = useState(null);

  // Health check
  useEffect(() => {
    fetch("http://localhost:5050/health")
      .then((r) => r.json())
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraOn(true);
      setError("");
    } catch {
      setError("📷 Could not access camera. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setFaces([]);
    setMessage("");
    if (intervalRef.current) clearInterval(intervalRef.current);
    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  };

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || analyzing) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.8);

    setAnalyzing(true);
    setError("");
    try {
      const res = await fetch("http://localhost:5050/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setFaces(data.faces || []);
        setMessage(data.message || "");
        setLastAnalyzedAt(new Date().toLocaleTimeString("en-IN"));
        drawOverlay(data.faces || [], canvas.width, canvas.height);
      }
    } catch {
      setError("❌ Could not connect to backend. Make sure app.py is running.");
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing]);

  const drawOverlay = (detectedFaces, w, h) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    overlay.width = w;
    overlay.height = h;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    detectedFaces.forEach((face) => {
      const { x, y, w: fw, h: fh } = face.bbox;
      const color = EMOTION_COLOR[face.emotion] || "#00FF88";
      const detPct = face.det_conf ? Math.round(face.det_conf * 100) : "";

      // Draw bounding box with glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, fw, fh);
      ctx.shadowBlur = 0;

      // Corner accents
      const cs = 14;
      ctx.lineWidth = 4;
      [[x,y],[x+fw,y],[x,y+fh],[x+fw,y+fh]].forEach(([cx,cy], i) => {
        ctx.beginPath();
        ctx.moveTo(cx + (i%2===0 ? cs : -cs), cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + (i<2 ? cs : -cs));
        ctx.stroke();
      });

      // Label strip
      const label = `${face.emotion}${detPct ? ` • 🎯${detPct}%` : ""} • ${face.gender}`;
      ctx.font = "bold 12px Inter, sans-serif";
      const tw = ctx.measureText(label).width + 14;
      ctx.fillStyle = color + "dd";
      ctx.fillRect(x, y - 26, tw, 26);
      ctx.fillStyle = "#000";
      ctx.fillText(label, x + 7, y - 8);
    });
  };

  // Auto-analyze every 3s
  useEffect(() => {
    if (autoMode && cameraOn) {
      intervalRef.current = setInterval(captureAndAnalyze, 3000);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [autoMode, cameraOn, captureAndAnalyze]);

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🎭</span>
            <div>
              <div className="logo-title">EmoTune <span className="logo-badge">Tamil</span></div>
              <div className="logo-sub">Face Emotion · Gender · Song Recommendations</div>
            </div>
          </div>
          <div className="backend-status">
            {backendOk === null && <span className="status-dot checking">⏳ Connecting...</span>}
            {backendOk === true && <span className="status-dot online">🟢 YOLOv8-face Online</span>}
            {backendOk === false && <span className="status-dot offline">🔴 Backend Offline</span>}
          </div>
        </div>
      </header>

      <main className="main">
        <div className="layout">
          {/* Left: Camera Panel */}
          <div className="camera-panel">
            <div className="camera-box">
              <video ref={videoRef} className="video-feed" playsInline muted />
              <canvas ref={overlayCanvasRef} className="overlay-canvas" />
              {!cameraOn && (
                <div className="camera-placeholder">
                  <div className="camera-placeholder-icon">📷</div>
                  <div>Enable camera to start</div>
                </div>
              )}
              {analyzing && (
                <div className="analyzing-badge">
                  <span className="spinner" /> Analyzing...
                </div>
              )}
              <canvas ref={canvasRef} style={{ display: "none" }} />
            </div>

            {/* Controls */}
            <div className="controls">
              {!cameraOn ? (
                <button
                  id="btn-start-camera"
                  className="btn btn-primary"
                  onClick={startCamera}
                  disabled={backendOk === false}
                >
                  📷 Start Camera
                </button>
              ) : (
                <button id="btn-stop-camera" className="btn btn-danger" onClick={stopCamera}>
                  ⏹ Stop Camera
                </button>
              )}

              <button
                id="btn-analyze"
                className="btn btn-accent"
                onClick={captureAndAnalyze}
                disabled={!cameraOn || analyzing}
              >
                {analyzing ? "⏳ Analyzing..." : "🔍 Analyze Now"}
              </button>

              <button
                id="btn-auto-mode"
                className={`btn ${autoMode ? "btn-warning" : "btn-secondary"}`}
                onClick={() => setAutoMode((v) => !v)}
                disabled={!cameraOn}
              >
                {autoMode ? "⏸ Stop Auto" : "▶ Auto Mode (3s)"}
              </button>
            </div>

            {lastAnalyzedAt && (
              <div className="last-analyzed">Last analyzed at {lastAnalyzedAt}</div>
            )}

            {error && <div className="alert alert-error">{error}</div>}
            {message && !faces.length && (
              <div className="alert alert-info">
                <span style={{ fontSize: "2rem" }}>😶</span>
                <div>{message}</div>
              </div>
            )}

            {/* Emotion Legend */}
            <div className="legend">
              <div className="legend-title">Emotion Legend</div>
              <div className="legend-grid">
                {Object.entries(EMOTION_EMOJI).map(([emotion, emoji]) => (
                  <div key={emotion} className="legend-item" style={{ background: EMOTION_COLOR[emotion] + "22" }}>
                    <span>{emoji}</span>
                    <span>{emotion}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Results Panel */}
          <div className="results-panel">
            {faces.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🎵</div>
                <div className="empty-title">No Analysis Yet</div>
                <div className="empty-sub">
                  Start camera &amp; click <strong>Analyze Now</strong> to detect your emotion and get
                  personalized Tamil song recommendations!
                </div>
                <div className="empty-tags">
                  <span>#TamilSongs</span>
                  <span>#EmotionDetection</span>
                  <span>#GenderAware</span>
                </div>
              </div>
            ) : (
              <div className="faces-container">
                <div className="results-header">
                  <span>🎭 Detected {faces.length} Face{faces.length > 1 ? "s" : ""}</span>
                  {autoMode && <span className="auto-badge">AUTO</span>}
                </div>
                {faces.map((face, i) => (
                  <FaceResult key={i} face={face} />
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        Built with ❤️ · Powered by TensorFlow + DeepFace · 🎵 Tamil Songs 2021–2025
      </footer>
    </div>
  );
}
