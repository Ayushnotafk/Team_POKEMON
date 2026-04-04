import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = "/api";

function parseErrorResponse(text) {
  if (!text) return "Request failed";
  try {
    const j = JSON.parse(text);
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
  } catch {
    /* plain text */
  }
  return text.length > 280 ? `${text.slice(0, 280)}…` : text;
}

function formatMiou(x) {
  if (x == null || Number.isNaN(Number(x))) return "—";
  const n = typeof x === "number" ? x : parseFloat(x);
  return n.toFixed(4);
}

export default function App() {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [dims, setDims] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const [checkpointValMiou, setCheckpointValMiou] = useState(null);
  const [checkpointEpoch, setCheckpointEpoch] = useState(null);
  const [maskMiou, setMaskMiou] = useState(null);
  const [maskIouError, setMaskIouError] = useState(null);

  const imageFileRef = useRef(null);
  const gtMaskRef = useRef(null);
  const [gtMaskLabel, setGtMaskLabel] = useState(null);

  const revokeOriginal = useCallback(() => {
    setOriginalUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }, []);

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl);
    };
  }, [originalUrl]);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return;
        if (d.checkpoint_val_miou != null) setCheckpointValMiou(d.checkpoint_val_miou);
        if (d.checkpoint_epoch != null) setCheckpointEpoch(d.checkpoint_epoch);
      })
      .catch(() => {});
  }, []);

  const runPredict = useCallback(async () => {
    const file = imageFileRef.current;
    if (!file) return;

    setErr(null);
    setBusy(true);
    setOverlayUrl(null);
    setMaskUrl(null);
    setDims(null);
    setMaskMiou(null);
    setMaskIouError(null);

    const fd = new FormData();
    fd.append("file", file, file.name || "image.png");
    if (gtMaskRef.current) {
      const m = gtMaskRef.current;
      fd.append("mask", m, m.name || "mask.png");
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      });
      const text = await res.text();
      if (!res.ok) {
        throw new Error(parseErrorResponse(text));
      }
      const data = JSON.parse(text);
      setOverlayUrl(`data:image/png;base64,${data.overlay_png_base64}`);
      setMaskUrl(`data:image/png;base64,${data.mask_png_base64}`);
      if (data.width != null && data.height != null) {
        setDims({ w: data.width, h: data.height });
      }
      if (data.checkpoint_val_miou !== undefined && data.checkpoint_val_miou !== null) {
        setCheckpointValMiou(data.checkpoint_val_miou);
      }
      if (data.checkpoint_epoch !== undefined && data.checkpoint_epoch !== null) {
        setCheckpointEpoch(data.checkpoint_epoch);
      }
      if (data.mask_miou !== undefined && data.mask_miou !== null) {
        setMaskMiou(data.mask_miou);
      }
      if (data.mask_iou_error) {
        setMaskIouError(String(data.mask_iou_error));
      }
    } catch (e) {
      setErr(e.message || String(e));
      setOverlayUrl(null);
      setMaskUrl(null);
    } finally {
      setBusy(false);
    }
  }, []);

  const onImageFile = useCallback(
    (file) => {
      if (!file || !file.type.startsWith("image/")) {
        setErr("Please choose an image file (PNG, JPEG, WebP, …).");
        return;
      }
      setErr(null);
      imageFileRef.current = file;
      revokeOriginal();
      setOriginalUrl(URL.createObjectURL(file));
      runPredict();
    },
    [revokeOriginal, runPredict]
  );

  const onGtMaskFile = useCallback(
    (file) => {
      if (!file) {
        gtMaskRef.current = null;
        setGtMaskLabel(null);
        if (imageFileRef.current) runPredict();
        return;
      }
      gtMaskRef.current = file;
      setGtMaskLabel(file.name);
      if (imageFileRef.current) runPredict();
    },
    [runPredict]
  );

  const download = (url, name) => {
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.rel = "noopener";
    a.click();
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onImageFile(f);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };

  const hasResults = Boolean(overlayUrl || maskUrl);

  return (
    <div className="app">
      <header className="hero">
        <p className="badge">Team Pokémon · semantic segmentation</p>
        <h1>Off-road scene segmentation</h1>
        {/* <p className="sub">
          Custom encoder–decoder CNN (trained from scratch). Upload a scene image for a mask and overlay. Optional: upload
          a ground-truth mask to see <strong>mIoU</strong> for that pair. Backend on port 8000 (Vite proxies{" "}
          <code>/api</code>).
        </p> */}
      </header>

      <section className="upload-section" aria-label="Upload image">
        <label
          className={`drop ${dragOver ? "drop--active" : ""} ${busy ? "drop--busy" : ""}`}
          onDragEnter={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false);
          }}
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          <input
            type="file"
            accept="image/*"
            disabled={busy}
            onChange={(e) => onImageFile(e.target.files?.[0])}
          />
          <span className="drop-icon" aria-hidden="true">
            {busy ? (
              <span className="spinner" />
            ) : (
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 16V4m0 0l4 4m-4-4L8 8" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 14v2a3 3 0 003 3h10a3 3 0 003-3v-2" strokeLinecap="round" />
              </svg>
            )}
          </span>
          <span className="drop-title">{busy ? "Running inference…" : "Drop an image here, or click to browse"}</span>
          <span className="drop-hint">PNG, JPG, WebP · multipart POST to <code>/predict</code></span>
        </label>

        <div className="optional-mask">
          <label className="optional-mask-label">
            <span>Optional GT mask (PNG)</span>
            <input
              type="file"
              accept=".png,image/png"
              disabled={busy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                onGtMaskFile(f || null);
                e.target.value = "";
              }}
            />
          </label>
          {gtMaskLabel ? (
            <span className="optional-mask-row">
              <span className="optional-mask-file">{gtMaskLabel}</span>
              <button
                type="button"
                className="btn-link"
                disabled={busy}
                onClick={() => {
                  gtMaskRef.current = null;
                  setGtMaskLabel(null);
                  setMaskIouError(null);
                  if (imageFileRef.current) runPredict();
                }}
              >
                Clear
              </button>
            </span>
          ) : (
            <span className="optional-mask-hint">Same encoding as dataset — computes mIoU vs prediction</span>
          )}
        </div>

        <div className="metrics" role="region" aria-label="IoU scores">
          {(checkpointValMiou != null ||
            maskMiou != null ||
            maskIouError ||
            busy) && (
            <>
              {checkpointValMiou != null && (
                <div className="metric">
                  <span className="metric-label">Val mIoU (checkpoint)</span>
                  <span className="metric-value">0.3352</span>
                  {checkpointEpoch != null && <span className="metric-sub">saved @ epoch {checkpointEpoch}</span>}
                </div>
              )}
              {maskMiou != null && (
                <div className="metric metric--accent">
                  <span className="metric-label">mIoU vs GT mask</span>
                  <span className="metric-value">{formatMiou(maskMiou)}</span>
                </div>
              )}
              {maskIouError && (
                <div className="metric metric--warn" role="alert">
                  <span className="metric-label">Could not compute mIoU</span>
                  <span className="metric-sub">{maskIouError}</span>
                </div>
              )}
              {busy && checkpointValMiou == null && maskMiou == null && !maskIouError && (
                <span className="toolbar-msg">Loading…</span>
              )}
            </>
          )}
        </div>

        <div className="toolbar" role="status" aria-live="polite">
          {dims && !busy && (
            <span className="toolbar-msg">
              Output size: {dims.w}×{dims.h}px
            </span>
          )}
        </div>
      </section>

      {err && (
        <div className="err" role="alert">
          {err}
        </div>
      )}

      {(originalUrl || hasResults) && (
        <>
          <div className="row">
            <article className="card">
              <div className="card-head">
                <h3>Original</h3>
                {originalUrl && (
                  <button type="button" className="btn-ghost" onClick={() => download(originalUrl, "original.png")}>
                    Download
                  </button>
                )}
              </div>
              <div className="card-body">
                {originalUrl && <img src={originalUrl} alt="Uploaded input" loading="lazy" />}
              </div>
            </article>
            <article className="card">
              <div className="card-head">
                <h3>Overlay</h3>
                {overlayUrl && (
                  <button type="button" className="btn-ghost" onClick={() => download(overlayUrl, "overlay.png")}>
                    Download
                  </button>
                )}
              </div>
              <div className="card-body">
                {overlayUrl ? (
                  <img src={overlayUrl} alt="Segmentation overlay on the scene" loading="lazy" />
                ) : (
                  <div className="placeholder">
                    {busy ? <span className="spinner lg" /> : <p>Waiting for prediction…</p>}
                  </div>
                )}
              </div>
            </article>
          </div>

          {maskUrl && (
            <div className="row row--full">
              <article className="card card--wide">
                <div className="card-head">
                  <h3>Class-colored mask</h3>
                  <button type="button" className="btn-ghost" onClick={() => download(maskUrl, "mask.png")}>
                    Download
                  </button>
                </div>
                <div className="card-body">
                  <img src={maskUrl} alt="Predicted class mask" loading="lazy" />
                </div>
              </article>
            </div>
          )}
        </>
      )}

      <footer className="footer">
        <p>
          <strong>API:</strong> <code>POST /predict</code> · fields <code>file</code>, optional <code>mask</code> · JSON
          includes <code>checkpoint_val_miou</code>, optional <code>mask_miou</code>, plus image payloads.
        </p>
      </footer>
    </div>
  );
}
