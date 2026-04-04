import { useCallback, useState } from "react";

const API_BASE = "/api";

export default function App() {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const onFile = useCallback(async (file) => {
    if (!file) return;
    setErr(null);
    setBusy(true);
    setOverlayUrl(null);
    setMaskUrl(null);
    const localUrl = URL.createObjectURL(file);
    setOriginalUrl(localUrl);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }
      const data = await res.json();
      setOverlayUrl(`data:image/png;base64,${data.overlay_png_base64}`);
      setMaskUrl(`data:image/png;base64,${data.mask_png_base64}`);
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="app">
      <h1>Off-road semantic segmentation</h1>
      <p className="sub">
        Custom CNN encoder–decoder (no pretrained weights). Upload a scene image; the API returns a colored class mask and an overlay.
      </p>

      <label className="drop">
        <input
          type="file"
          accept="image/*"
          disabled={busy}
          onChange={(e) => onFile(e.target.files?.[0])}
        />
        {busy ? "Running inference…" : "Click or drop an image here"}
      </label>

      {err && <div className="err">{err}</div>}

      {(originalUrl || overlayUrl) && (
        <div className="row">
          <div className="card">
            <h3>Original</h3>
            {originalUrl && (
              <img src={originalUrl} alt="Original" />
            )}
          </div>
          <div className="card">
            <h3>Segmentation overlay</h3>
            {overlayUrl ? (
              <img src={overlayUrl} alt="Overlay" />
            ) : (
              <p className="meta">Waiting for prediction…</p>
            )}
          </div>
        </div>
      )}

      {maskUrl && (
        <div className="row" style={{ marginTop: "1rem" }}>
          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h3>Class-colored mask</h3>
            <img src={maskUrl} alt="Mask" />
          </div>
        </div>
      )}

      <p className="meta">
        Backend: POST /predict (multipart file). Ensure the FastAPI server is running on port 8000 so the Vite proxy can reach it.
      </p>
    </div>
  );
}
