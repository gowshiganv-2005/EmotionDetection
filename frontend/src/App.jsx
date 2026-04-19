import React, { useRef, useState, useEffect, useCallback } from "react";
import "./App.css";

const EMOTION_EMOJI = {
  Happy: "😊",
  Sad: "😔",
  Angry: "😠",
  Fear: "😨",
  Surprise: "😮",
  Disgust: "😒",
  Neutral: "😐",
};

const EMOTION_THEME = {
  Happy: "#88D4AB",    // Celadon
  Sad: "#67B99A",      // Mint
  Angry: "#469D89",    // Zomp
  Fear: "#248277",     // Pine Green
  Surprise: "#88D4AB", // Celadon
  Disgust: "#036666",  // Caribbean Current
  Neutral: "#469D89",  // Zomp
};

function MusicItem({ song, index, onPlay }) {
  return (
    <a 
      href={song.youtube} 
      target="_blank" 
      rel="noopener noreferrer" 
      className="music-item"
      style={{ animationDelay: `${index * 0.1}s` }}
      onClick={onPlay}
    >
      <div className="album-art"></div>
      <div className="track-info">
        <div className="track-name">{song.title}</div>
        <div className="artist-name">{song.singer} • {song.movie}</div>
      </div>
      <div className="play-icon"></div>
    </a>
  );
}

function ResultCard({ face, onPlaySong }) {
  const theme = EMOTION_THEME[face.emotion] || "#6366f1";
  
  return (
    <div className="glass-card emotion-card" style={{ borderTop: `4px solid ${theme}` }}>
      <div className="user-profile">
        <div className="emotion-emoji-grid" style={{ boxShadow: `0 10px 20px -5px ${theme}66`, color: theme }}>
          {EMOTION_EMOJI[face.emotion] || "😐"}
        </div>
        <div className="user-meta">
          <h2 style={{ color: theme, fontSize: '1.4rem', fontWeight: 800 }}>{face.emotion}</h2>
          <div className="status-pills">
            <span className="pill"> YOLO {Math.round(face.det_conf * 100)}%</span>
            <span className="pill">{face.gender === 'Woman' ? '👩' : '👨'} {face.gender}</span>
            <span className="pill">Age {face.age}</span>
          </div>
        </div>
      </div>
      
      <div className="music-section">
        <h3 className="result-heading" style={{ marginBottom: '1rem', fontSize: '1rem' }}>
           Recommended Tamil Tracks
        </h3>
        <div className="music-list">
          {face.songs.map((song, i) => (
            <MusicItem key={i} song={song} index={i} onPlay={onPlaySong} />
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
  const autoSyncInterval = useRef(null);
  
  const [cameraOn, setCameraOn] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [isAutoSync, setIsAutoSync] = useState(false);
  const [faces, setFaces] = useState([]);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Checking...");

  useEffect(() => {
    fetch("/health")
      .then(r => r.json())
      .then(d => setStatus(d.detector || "Online"))
      .catch(() => setStatus("Offline"));
  }, []);

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setCameraOn(true);
      setError("");
    } catch (e) {
      setError("Camera access denied. Please check permissions.");
    }
  };

  const handleStop = () => {
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach(t => t.stop());
    videoRef.current.srcObject = null;
    setCameraOn(false);
    setIsAutoSync(false);
    setFaces([]);
    if (autoSyncInterval.current) clearInterval(autoSyncInterval.current);
  };

  const analyze = useCallback(async () => {
    if (!videoRef.current || analyzing) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    
    setAnalyzing(true);
    try {
      const res = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.8) }),
      });
      const data = await res.json();
      if (data.faces) {
        setFaces(data.faces);
        drawOverlay(data.faces);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setAnalyzing(false);
    }
  }, [analyzing]);

  // Handle Play Click: Starts Sync automatically
  const onPlaySong = () => {
    if (!isAutoSync) {
      setIsAutoSync(true);
    }
  };

  // Auto-Sync Logic
  useEffect(() => {
    if (isAutoSync && cameraOn) {
      autoSyncInterval.current = setInterval(() => {
        analyze();
      }, 3000); // 3-second sync
    } else {
      if (autoSyncInterval.current) clearInterval(autoSyncInterval.current);
    }
    return () => {
      if (autoSyncInterval.current) clearInterval(autoSyncInterval.current);
    };
  }, [isAutoSync, cameraOn, analyze]);

  const drawOverlay = (detected) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay || !videoRef.current) return;
    const cw = videoRef.current.videoWidth;
    const ch = videoRef.current.videoHeight;
    overlay.width = cw;
    overlay.height = ch;
    const ctx = overlay.getContext("2d");
    
    detected.forEach(f => {
      const { x, y, w, h } = f.bbox;
      const color = EMOTION_THEME[f.emotion] || "#67B99A";
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);
      
      ctx.fillStyle = color;
      ctx.font = "bold 16px Inter";
      ctx.fillText(`${f.emotion} ${Math.round(f.det_conf*100)}%`, x, y - 10);
    });
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <div className="logo-container">🎭</div>
            <div className="brand-text">
              <h1>RajGowthamTune <span className="badge-tamil">Tamil</span></h1>
            </div>
          </div>
          <div className="status-pills">
            {isAutoSync && <span className="pill" style={{ background: 'var(--c2)', color: '#011a1a', border: 'none' }}>⚡ AUTO-SYNC ACTIVE</span>}
            <div className={`pill ${status === 'Offline' ? 'error' : 'success'}`} style={{ color: status === 'Offline' ? '#f87171' : '#10b981' }}>
              ● {status}
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="scanner-container">
          <div className="glass-card">
            <div className="camera-viewport">
              <video ref={videoRef} className="video-element" muted playsInline />
              <canvas ref={overlayCanvasRef} style={{ position: "absolute", inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
              
              {!cameraOn && (
                <div className="empty-state">
                  <i style={{ fontSize: '3.5rem', display: 'block', marginBottom: '1rem' }}>📷</i>
                  <p>Ready to Scan. Please enable your camera.</p>
                </div>
              )}
              
              {cameraOn && (
                <>
                  <div className="scanner-bracket tl"></div>
                  <div className="scanner-bracket tr"></div>
                  <div className="scanner-bracket bl"></div>
                  <div className="scanner-bracket br"></div>
                  <div className="scan-line"></div>
                </>
              )}
              
              <canvas ref={canvasRef} style={{ display: "none" }} />
            </div>
            
            <div className="action-bar">
              {!cameraOn ? (
                <button className="btn-premium primary" onClick={handleStart}>
                  Start Camera
                </button>
              ) : (
                <>
                  <button className="btn-premium" onClick={handleStop}>Stop</button>
                  <button className="btn-premium primary" onClick={analyze} disabled={analyzing}>
                    {analyzing ? "Scanning..." : "Sync Emotion"}
                  </button>
                  <button 
                    className={`btn-premium ${isAutoSync ? 'active' : ''}`} 
                    onClick={() => setIsAutoSync(!isAutoSync)}
                    style={isAutoSync ? { background: 'var(--c4)', border: '1px solid var(--c1)' } : {}}
                  >
                    {isAutoSync ? "Pause Sync" : "Auto Sync"}
                  </button>
                </>
              )}
            </div>
          </div>
          
          {error && <div className="pill" style={{ color: '#f87171', border: '1px solid currentColor', width: 'fit-content' }}>{error}</div>}
        </section>

        <aside className="results-container">
          <div className="result-heading">
            <span></span> Live Experience Analytics
          </div>
          
          {faces.length === 0 ? (
            <div className="glass-card empty-state" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <i style={{ fontSize: '4.5rem', display: 'block', marginBottom: '1.5rem' }}></i>
              <h3 style={{ fontSize: '1.6rem' }}>No Analysis Detected</h3>
              <p style={{ color: 'var(--text-dim)', fontSize: '1rem', marginTop: '1rem', lineHeight: '1.8', maxWidth: '360px', margin: '0 auto' }}>
                Capture a face to see real-time emotion telemetry and local song matches.
              </p>
            </div>
          ) : (
            faces.map((f, i) => <ResultCard key={i} face={f} onPlaySong={onPlaySong} />)
          )}
        </aside>
      </main>
    </div>
  );
}
