import { useEffect, useState } from "react";
import { apiUrl, type CameraStatus, type EventItem } from "../api";
import { cameraHealth, fmtAgo, fmtClock } from "../lib/charts";

export function StatusBadge({ camera }: { camera: { state: string; stalled?: boolean } | undefined }) {
  const h = cameraHealth(camera);
  return (
    <span className={`status ${h.level}`}>
      <span className="dot" aria-hidden="true">{h.icon}</span>
      {h.label}
    </span>
  );
}

interface FrameInfo {
  seq: number;
  ageMs: number | null;
  received: number;
}

/** Live frames by long polling: ask for "a frame newer than the one I have"; the server answers when the
 *  camera's next processed frame exists (or 204 after 2 s). Pauses while the tab is hidden. */
function useLiveFrame(cameraId: string): { src: string | null; info: FrameInfo | null; error: string | null } {
  const [src, setSrc] = useState<string | null>(null);
  const [info, setInfo] = useState<FrameInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    let seq = 0;
    let url: string | null = null;
    const ctrl = new AbortController();
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    (async () => {
      while (!stop) {
        if (document.visibilityState !== "visible") {
          await wait(500);
          continue;
        }
        try {
          const r = await fetch(apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/frame.jpg`, { after: seq, timeout: 2 }),
            { signal: ctrl.signal, cache: "no-store" });
          if (r.status === 204) continue;
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const blob = await r.blob();
          seq = Number(r.headers.get("X-Frame-Seq") ?? seq);
          const next = URL.createObjectURL(blob);
          if (url) URL.revokeObjectURL(url);
          url = next;
          if (stop) break;
          setSrc(next);
          setInfo({ seq, ageMs: Number(r.headers.get("X-Frame-Age-Ms")), received: Date.now() });
          setError(null);
        } catch (e) {
          if (stop) break;
          setError(e instanceof Error ? e.message : String(e));
          await wait(1000);
        }
      }
    })();
    return () => {
      stop = true;
      ctrl.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [cameraId]);
  return { src, info, error };
}

function CameraTile({ id, cam }: { id: string; cam: CameraStatus | undefined }) {
  const { src, info, error } = useLiveFrame(id);
  return (
    <div className="card cam">
      <div className="frame">
        {src ? <img src={src} alt={`Camera ${id}, latest processed frame with tracked objects`} />
             : <span className="muted">{error ? `No frames (${error})` : "Waiting for the first frame…"}</span>}
      </div>
      <div className="meta">
        <b>{id}</b>
        <StatusBadge camera={cam} />
        <span className="num">{cam?.processed_fps != null ? `${cam.processed_fps.toFixed(1)} fps` : "–"}</span>
        <span className="num" title="frame decoded → tracks ready (median)">
          {cam?.latency_ms_p50 != null ? `${Math.round(cam.latency_ms_p50)} ms` : ""}
        </span>
        <span className="num">{cam?.active_tracks ?? 0} tracks</span>
        <span className="grow" />
        <span className="num muted" title="age of the shown frame when it was sent">
          {info?.ageMs ? `frame age ${Math.round(info.ageMs)} ms` : ""}
        </span>
      </div>
    </div>
  );
}

function describe(e: EventItem): { what: string; detail: string } {
  if (e.type === "identity_link") {
    const d = e.data as { global_id?: number; camera?: string; from_cameras?: string[]; score?: number };
    return {
      what: `G${d.global_id} seen in ${d.camera}`,
      detail: `after ${(d.from_cameras ?? []).join(", ")} · similarity ${d.score?.toFixed(2) ?? "–"}`,
    };
  }
  if (e.type === "camera_state") {
    const d = e.data as { from?: string; to?: string; error?: string | null };
    return { what: `${e.camera_id}: ${d.from} → ${d.to}`, detail: d.error ?? "" };
  }
  return { what: e.type, detail: "" };
}

export function LiveView({ cameras, events, onIdentity }: {
  cameras: Record<string, CameraStatus> | undefined;
  events: EventItem[];
  onIdentity: (id: number) => void;
}) {
  const ids = Object.keys(cameras ?? {}).sort();
  return (
    <div className="live">
      <div className="cams">
        {ids.length === 0 && <div className="card empty">No live cameras (replay mode, or the server is starting).</div>}
        {ids.map((id) => <CameraTile key={id} id={id} cam={cameras?.[id]} />)}
      </div>
      <div className="card feed">
        <div className="card-h"><h2>Events</h2><span className="sub">cross-camera links, camera state</span></div>
        <div className="card-b" style={{ padding: 0, marginTop: 8 }}>
          {events.length === 0 ? <div className="empty">No events yet.</div> : (
            <ul>
              {events.map((e, i) => {
                const d = describe(e);
                const body = (
                  <>
                    <div className="what">{d.what}</div>
                    <div className="when">{fmtClock(e.t)} · {fmtAgo(e.t)}{d.detail ? ` · ${d.detail}` : ""}</div>
                  </>
                );
                return (
                  <li key={e.id ?? `${e.type}-${e.t}-${i}`}>
                    {e.identity_id ? <button onClick={() => onIdentity(e.identity_id!)}>{body}</button> : body}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
