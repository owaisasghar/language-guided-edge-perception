import { useEffect, useState } from "react";
import { apiUrl, getJSON, thumbUrl, type CameraStatus, type EventItem, type Status, type SystemInfo } from "../api";
import { useInterval, useNow, type Sample } from "../hooks";
import { cameraHealth, fmtAgo, fmtClock } from "../lib/charts";
import { IconAlert, IconArrowRight, IconCamera, IconCheck, IconExpand, IconGrid, IconLink, IconOffline } from "./Icons";
import { KpiStrip } from "./Kpis";
import { PageHead } from "./Layout";

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
          if (!r.ok) throw new Error(r.status === 409 ? "replay mode" : `HTTP ${r.status}`);
          const blob = await r.blob();
          seq = Number(r.headers.get("X-Frame-Seq") ?? seq);
          const next = URL.createObjectURL(blob);
          if (url) URL.revokeObjectURL(url);
          url = next;
          if (stop) break;
          setSrc(next);
          setInfo({ seq, ageMs: Number(r.headers.get("X-Frame-Age-Ms")) });
          setError(null);
        } catch (e) {
          if (stop) break;
          setError(e instanceof Error ? e.message : String(e));
          await wait(1500);
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

function Frame({ id, cam, large = false }: { id: string; cam: CameraStatus | undefined; large?: boolean }) {
  const { src, info, error } = useLiveFrame(id);
  const streaming = cam?.state === "streaming" && !cam?.stalled;
  return (
    <div className="frame" style={large ? { aspectRatio: "16 / 9" } : undefined}>
      {src ? <img src={src} alt={`Camera ${id}: latest processed frame with tracked objects`} />
           : <div className="placeholder"><IconCamera size={28} />{error ? `No frames (${error})` : "Waiting for the first frame…"}</div>}
      <div className="ov top">
        <span className="name">{id}</span>
        <span className="grow" />
        <span className={`live-pill${streaming ? "" : " off"}`}><i />{streaming ? "LIVE" : cameraHealth(cam).label.toUpperCase()}</span>
      </div>
      <div className="ov bottom">
        <span className="stat">{cam?.processed_fps != null ? `${cam.processed_fps.toFixed(1)} fps` : "– fps"}</span>
        <span className="stat" title="frame decoded → tracks ready (median)">
          {cam?.latency_ms_p50 != null ? `${Math.round(cam.latency_ms_p50)} ms` : "– ms"}
        </span>
        <span className="stat">{cam?.active_tracks ?? 0} tracked</span>
        <span className="grow" />
        {info?.ageMs ? <span className="stat" title="age of this frame when it was sent">{Math.round(info.ageMs)} ms old</span> : null}
      </div>
    </div>
  );
}

function CameraTile({ id, cam, onOpen }: { id: string; cam: CameraStatus | undefined; onOpen: () => void }) {
  return (
    <button className="cam" onClick={onOpen} aria-label={`Open camera ${id}`}>
      <Frame id={id} cam={cam} />
      <div className="cam-meta">
        <b>{id}</b>
        <StatusBadge camera={cam} />
        <span className="grow" />
        <span className="muted num">{cam?.input_fps != null ? `${cam.input_fps.toFixed(0)} fps in` : ""}</span>
        <IconExpand size={15} />
      </div>
    </button>
  );
}

interface LiveTracks {
  tracks: { key: string; local_id: number; category: string; score: number; provisional_global_id: number | null }[];
}

function FocusView({ id, cam, onBack }: { id: string; cam: CameraStatus | undefined; onBack: () => void }) {
  return (
    <div className="card" style={{ overflow: "hidden" }}>
      <Frame id={id} cam={cam} large />
      <div className="cam-meta">
        <b>{id}</b><StatusBadge camera={cam} />
        <span className="muted mono">{cam?.url}</span>
        <span className="grow" />
        <button className="btn-ghost" onClick={onBack}><IconGrid size={15} />All cameras</button>
      </div>
    </div>
  );
}

function InView({ id }: { id: string }) {
  const [live, setLive] = useState<LiveTracks | null>(null);
  useInterval(() => {
    getJSON<LiveTracks>(`/api/cameras/${encodeURIComponent(id)}/tracks`).then(setLive).catch(() => setLive(null));
  }, 1000, [id]);
  const tracks = [...(live?.tracks ?? [])].sort((a, b) => (b.provisional_global_id != null ? 1 : 0) - (a.provisional_global_id != null ? 1 : 0) || a.local_id - b.local_id);
  return (
    <div className="feed">
      <section className="card" style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div className="card-h"><div className="grow"><h2>In view now</h2><div className="sub">{tracks.length} tracked on {id} · blue: a likely identity</div></div></div>
        <div className="track-list" style={{ marginTop: 10 }}>
          {tracks.length === 0 && <div className="empty-state" style={{ padding: 24 }}>Nothing tracked at the moment.</div>}
          {tracks.map((t) => (
            <div key={t.key} className={`track-li${t.provisional_global_id != null ? " prov" : ""}`}>
              <span className="sw" aria-hidden="true" />
              <div className="grow">
                <div><b>#{t.local_id}</b> <span className="secondary">{t.category}</span></div>
                <div className="muted num" style={{ fontSize: 12 }}>detector score {t.score.toFixed(2)}</div>
              </div>
              {t.provisional_global_id != null
                ? <span className="gid" title="the identity this track would join if it ended now">G{t.provisional_global_id}?</span>
                : <span className="muted" style={{ fontSize: 12 }}>new</span>}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function FeedItem({ e, now, onIdentity }: { e: EventItem; now: number; onIdentity: (id: number) => void }) {
  if (e.type === "identity_link") {
    const d = e.data as { global_id?: number; camera?: string; from_cameras?: string[]; score?: number };
    return (
      <li className="feed-item">
        <button className="feed-btn" onClick={() => e.identity_id && onIdentity(e.identity_id)}>
          {e.track_id ? <img className="feed-thumb" src={thumbUrl(e.track_id)} alt="" loading="lazy" onError={(ev) => (ev.currentTarget.style.visibility = "hidden")} />
                      : <span className="feed-icon"><IconLink size={16} /></span>}
          <div className="grow">
            <div className="feed-what"><span className="gid">G{d.global_id}</span> seen in <span className="cam-chip">{d.camera}</span></div>
            <div className="feed-detail">after {(d.from_cameras ?? []).join(", ")} · similarity {d.score?.toFixed(2) ?? "–"}</div>
            <div className="feed-when">{fmtClock(e.t)} · {fmtAgo(e.t, now)}</div>
          </div>
          <IconArrowRight size={15} />
        </button>
      </li>
    );
  }
  const d = e.data as { from?: string; to?: string; error?: string | null };
  const h = cameraHealth({ state: (d.to ?? "").split("/")[0], stalled: (d.to ?? "").includes("stalled") });
  return (
    <li className="feed-item">
      <div className="feed-static">
        <span className="feed-icon" style={{ color: `var(--${h.level === "good" ? "good-text" : h.level})` }}>
          {h.level === "good" ? <IconCheck size={16} /> : h.level === "critical" ? <IconOffline size={16} /> : <IconAlert size={16} />}
        </span>
        <div className="grow">
          <div className="feed-what">{e.camera_id}: {h.label.toLowerCase()}</div>
          <div className="feed-detail">{d.from} → {d.to}{d.error ? ` · ${d.error}` : ""}</div>
          <div className="feed-when">{fmtClock(e.t)} · {fmtAgo(e.t, now)}</div>
        </div>
      </div>
    </li>
  );
}

export function LiveView({ status, history, events, closedTracks, system, onIdentity }: {
  status: Status | null; history: Sample[]; events: EventItem[]; closedTracks: number; system: SystemInfo | null;
  onIdentity: (id: number) => void;
}) {
  const [focus, setFocus] = useState<string | null>(() => new URLSearchParams(window.location.search).get("camera")); // ?camera=cam02
  const now = useNow(5000);
  const cameras = status?.cameras;
  const ids = Object.keys(cameras ?? {}).sort();
  const det = system?.detector;
  const sub = status?.live === false ? "Replay of stored runs: no live cameras."
    : `${ids.length} camera${ids.length === 1 ? "" : "s"}${det ? ` · ${det.model_id ?? "detector"} (${det.dtype})` : ""}`
      + `${system?.pipeline ? ` · looking for ${system.pipeline.phrases.join(", ")}` : ""}`;
  return (
    <>
      <PageHead title="Live" sub={sub} actions={focus && (
        <div className="seg" role="group" aria-label="Layout">
          <button aria-pressed={false} onClick={() => setFocus(null)}><IconGrid size={14} />Grid</button>
          <button aria-pressed={true}><IconExpand size={14} />{focus}</button>
        </div>
      )} />
      <KpiStrip status={status} history={history} closedTracks={closedTracks} />
      <div className="live">
        <div>
          {focus && cameras?.[focus] ? <FocusView id={focus} cam={cameras[focus]} onBack={() => setFocus(null)} /> : (
            <div className={`cams${ids.length > 4 ? " three" : ""}`}>
              {status && ids.length === 0 && (
                <div className="card empty-state" style={{ gridColumn: "1 / -1" }}>
                  <div className="art"><IconCamera size={28} /></div>
                  <h3>No live cameras</h3>
                  <div>This server is replaying stored runs. Search and the timeline work as usual.</div>
                </div>
              )}
              {!status && [0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ aspectRatio: "16 / 10" }} />)}
              {ids.map((id) => <CameraTile key={id} id={id} cam={cameras?.[id]} onOpen={() => setFocus(id)} />)}
            </div>
          )}
        </div>
        {focus && cameras?.[focus] ? <InView id={focus} /> : <aside className="card feed" aria-label="Activity">
          <div className="card-h"><div className="grow"><h2>Activity</h2><div className="sub">cross-camera links and camera health</div></div>
            <span className="chip num">{events.length}</span></div>
          <div style={{ marginTop: 10, overflow: "auto" }}>
            {events.length === 0 ? (
              <div className="empty-state" style={{ padding: 28 }}>
                <div className="art" style={{ width: 48, height: 48 }}><IconLink size={22} /></div>
                <div>No events yet. When an identity is seen by a second camera, it appears here.</div>
              </div>
            ) : (
              <ul className="feed-list">
                {events.map((e, i) => <FeedItem key={e.id ?? `${e.type}-${e.t}-${i}`} e={e} now={now} onIdentity={onIdentity} />)}
              </ul>
            )}
          </div>
        </aside>}
      </div>
    </>
  );
}
