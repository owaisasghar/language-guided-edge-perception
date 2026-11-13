import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { getJSON, type Timeline, type Track } from "../api";
import { bucketSeconds, compact, fmtClock, fmtDuration, packRows, seqStep, timeTicks } from "../lib/charts";

const RANGES: [string, number][] = [["5 min", 300], ["15 min", 900], ["1 h", 3600], ["6 h", 21600], ["24 h", 86400]];
const SWIMLANE_MAX_S = 1800; // individual tracks only for windows up to 30 min
const LABEL_W = 64;
const ROW_H = 6; // track bar height; 2 px surface gap between rows
const MAX_ROWS = 18;

interface Tip { x: number; y: number; body: ReactNode }

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  const left = Math.min(tip.x + 14, window.innerWidth - 240);
  return <div className="tooltip" style={{ left, top: tip.y + 14 }}>{tip.body}</div>;
}

function useWidth(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function TimeAxis({ since, until, x, y, height }: { since: number; until: number; x: (t: number) => number; y: number; height: number }) {
  const ticks = timeTicks(since, until, 6);
  const withSeconds = until - since <= 1800;
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={height} stroke="var(--grid)" strokeWidth={1} />
          <text x={x(t)} y={y} textAnchor="middle" className="num">{fmtClock(t, withSeconds)}</text>
        </g>
      ))}
    </g>
  );
}

// --------------------------------------------------------------------------------------- density heatmap
function Density({ tl, cameras, setTip }: { tl: Timeline; cameras: string[]; setTip: (t: Tip | null) => void }) {
  const [ref, width] = useWidth();
  const [table, setTable] = useState(false);
  const n = tl.buckets.length;
  const cellW = (width - LABEL_W) / n;
  const cellH = 20;
  const max = Math.max(1, ...cameras.flatMap((c) => tl.tracks_started[c] ?? []));
  const maxLinks = Math.max(1, ...tl.identity_links);
  const linksH = 48;
  const heatH = cameras.length * (cellH + 2);
  const x = (t: number) => LABEL_W + ((t - tl.since) / (tl.until - tl.since)) * (width - LABEL_W);
  const total = heatH + 18 + linksH + 22;
  const range = (i: number) => `${fmtClock(tl.buckets[i])} – ${fmtClock(tl.buckets[i] + tl.bucket_s)}`;
  return (
    <div className="card">
      <div className="card-h">
        <h2>Tracks started per camera</h2>
        <span className="sub">per {fmtDuration(tl.bucket_s)}; below: cross-camera links</span>
        <span className="grow" />
        <span className="legend" aria-label="Colour scale">
          <span className="muted">fewer</span>
          {[1, 2, 3, 4, 5, 6, 7].map((s) => <span key={s} className="key" style={{ background: `var(--seq-${s})`, marginRight: 0 }} />)}
          <span className="muted">more (max {max})</span>
        </span>
      </div>
      <div className="card-b">
        <div className="chart" ref={ref}>
          <svg width={width} height={total} role="img" aria-label="Heatmap of tracks started per camera over time">
            {cameras.map((c, r) => (
              <g key={c} transform={`translate(0, ${r * (cellH + 2)})`}>
                <text className="lane-label" x={0} y={cellH / 2 + 4}>{c}</text>
                {(tl.tracks_started[c] ?? new Array(n).fill(0)).map((v, i) => {
                  const s = seqStep(v, max);
                  return (
                    <rect key={i} x={LABEL_W + i * cellW + 1} y={0} width={Math.max(1, cellW - 2)} height={cellH} rx={2}
                          fill={s ? `var(--seq-${s})` : "var(--surface-2)"}
                          onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, body: (
                            <><div className="v num">{v} track{v === 1 ? "" : "s"} started</div><div className="k">{c} · {range(i)}</div></>) })}
                          onMouseLeave={() => setTip(null)} />
                  );
                })}
              </g>
            ))}
            <g transform={`translate(0, ${heatH + 18})`}>
              <text className="lane-label" x={0} y={linksH / 2 + 4}>links</text>
              <line x1={LABEL_W} x2={width} y1={linksH} y2={linksH} stroke="var(--axis)" strokeWidth={1} />
              {tl.identity_links.map((v, i) => {
                const h = (v / maxLinks) * (linksH - 4);
                const bw = Math.min(24, Math.max(2, cellW - 2));
                const bx = LABEL_W + i * cellW + (cellW - bw) / 2;
                return (
                  <g key={i}>
                    {v > 0 && <path d={`M${bx},${linksH} v${-(h - 2)} q0,-2 2,-2 h${bw - 4} q2,0 2,2 v${h - 2} z`} fill="var(--series-1)" />}
                    <rect x={LABEL_W + i * cellW} y={0} width={cellW} height={linksH} fill="transparent"
                          onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, body: (
                            <><div className="v num">{v} cross-camera link{v === 1 ? "" : "s"}</div><div className="k">{range(i)}</div></>) })}
                          onMouseLeave={() => setTip(null)} />
                  </g>
                );
              })}
              <text x={width} y={-2} textAnchor="end">max {maxLinks}</text>
            </g>
            <g transform={`translate(0, ${heatH + 18 + linksH})`}>
              {timeTicks(tl.since, tl.until, 6).map((t) => (
                <text key={t} x={x(t)} y={16} textAnchor="middle" className="num">{fmtClock(t, tl.until - tl.since <= 1800)}</text>
              ))}
            </g>
          </svg>
        </div>
        <button className="icon-btn" onClick={() => setTable(!table)} aria-expanded={table}>{table ? "Hide table" : "Show as table"}</button>
        {table && (
          <div style={{ overflowX: "auto", marginTop: 8 }}>
            <table className="data">
              <thead><tr><th>from</th>{cameras.map((c) => <th className="n" key={c}>{c}</th>)}<th className="n">links</th></tr></thead>
              <tbody>
                {tl.buckets.map((b, i) => (
                  <tr key={b}><td className="num">{fmtClock(b)}</td>
                    {cameras.map((c) => <td className="n" key={c}>{tl.tracks_started[c]?.[i] ?? 0}</td>)}
                    <td className="n">{tl.identity_links[i]}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------- swimlanes
function Swimlanes({ tracks, cameras, since, until, setTip, onIdentity }: {
  tracks: Track[]; cameras: string[]; since: number; until: number; setTip: (t: Tip | null) => void; onIdentity: (id: number) => void;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);
  const x = useCallback((t: number) => LABEL_W + ((Math.min(Math.max(t, since), until) - since) / (until - since)) * (width - LABEL_W),
    [since, until, width]);
  const lanes = useMemo(() => cameras.map((c) => {
    const ts = tracks.filter((t) => t.camera_id === c);
    const { rows, count } = packRows(ts, 0.5);
    return { camera: c, tracks: ts, rows, count: Math.min(count, MAX_ROWS), hidden: rows.filter((r) => r >= MAX_ROWS).length };
  }), [tracks, cameras]);
  let y = 0;
  const laneY = lanes.map((l) => {
    const top = y;
    y += Math.max(1, l.count) * (ROW_H + 2) + 14;
    return top;
  });
  const height = y + 20;
  const multi = (t: Track) => t.identity_cameras.length > 1;
  return (
    <div className="card">
      <div className="card-h">
        <h2>Tracks per camera</h2>
        <span className="sub">each bar is one local track; hover to highlight its identity in every camera, click for details</span>
        <span className="grow" />
        <span className="legend">
          <span><span className="key" style={{ background: "var(--series-1)" }} />seen by several cameras</span>
          <span><span className="key" style={{ background: "var(--deemph)" }} />one camera</span>
        </span>
      </div>
      <div className="card-b">
        <div className="chart" ref={ref}>
          <svg width={width} height={height} role="img" aria-label="Swimlane chart of tracks per camera over time">
            <TimeAxis since={since} until={until} x={x} y={height - 4} height={height - 18} />
            {lanes.map((l, li) => (
              <g key={l.camera} transform={`translate(0, ${laneY[li]})`}>
                <text className="lane-label" x={0} y={10}>{l.camera}</text>
                {l.hidden > 0 && <text x={0} y={24}>+{l.hidden} more</text>}
                {l.tracks.map((t, i) => {
                  if (l.rows[i] >= MAX_ROWS) return null;
                  const x1 = x(t.t_start);
                  const w = Math.max(2, x(t.t_end) - x1);
                  const dim = hover != null && t.identity_id !== hover;
                  return (
                    <g key={t.id} style={{ cursor: t.identity_id ? "pointer" : "default" }}
                       onMouseMove={(e) => {
                         setHover(multi(t) ? t.identity_id : null);
                         setTip({ x: e.clientX, y: e.clientY, body: (
                           <>
                             <div className="v num">{fmtDuration(t.duration_s)}</div>
                             <div className="k">{t.key} · {t.category} · {fmtClock(t.t_start)}</div>
                             <div className="k">{t.global_id != null ? `G${t.global_id}: ${t.identity_cameras.join(", ")}` : "no identity yet"}</div>
                           </>) });
                       }}
                       onMouseLeave={() => { setHover(null); setTip(null); }}
                       onClick={() => t.identity_id && onIdentity(t.identity_id)}>
                      <rect x={x1} y={l.rows[i] * (ROW_H + 2)} width={w} height={ROW_H + 2} fill="transparent" />
                      <rect x={x1} y={l.rows[i] * (ROW_H + 2) + 1} width={w} height={ROW_H} rx={2}
                            fill={multi(t) ? "var(--series-1)" : "var(--deemph)"} opacity={dim ? 0.25 : 1} />
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
        </div>
        <div className="row">
          <button className="icon-btn" onClick={() => setTable(!table)} aria-expanded={table}>{table ? "Hide table" : "Show as table"}</button>
          <span className="muted">{compact(tracks.length)} tracks in this window</span>
        </div>
        {table && (
          <div style={{ maxHeight: 360, overflow: "auto", marginTop: 8 }}>
            <table className="data">
              <thead><tr><th>track</th><th>start</th><th className="n">duration</th><th>identity</th><th>cameras</th></tr></thead>
              <tbody>
                {tracks.map((t) => (
                  <tr key={t.id}><td>{t.key}</td><td className="num">{fmtClock(t.t_start)}</td><td className="n">{fmtDuration(t.duration_s)}</td>
                    <td>{t.global_id != null ? `G${t.global_id}` : "–"}</td><td>{t.identity_cameras.join(", ")}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ view
export function TimelineView({ cameras, live, onIdentity }: { cameras: string[]; live: boolean; onIdentity: (id: number) => void }) {
  const [span, setSpan] = useState(300);
  const [tl, setTl] = useState<Timeline | null>(null);
  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [end, setEnd] = useState<number | null>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      let until = Date.now() / 1000;
      if (!live) { // replay: end of the stored data
        const last = await getJSON<Track[]>("/api/tracks", { limit: 1 });
        until = last.length ? last[0].t_end : until;
      }
      const since = until - span;
      const [t, tr] = await Promise.all([
        getJSON<Timeline>("/api/timeline", { since, until, bucket_s: bucketSeconds(span) }),
        span <= SWIMLANE_MAX_S ? getJSON<Track[]>("/api/tracks", { since, until, limit: 5000 }) : Promise.resolve(null),
      ]);
      setTl(t);
      setTracks(tr);
      setEnd(until);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [span, live]);

  useEffect(() => {
    load();
    if (!live) return;
    const id = window.setInterval(() => document.visibilityState === "visible" && load(), 5000);
    return () => window.clearInterval(id);
  }, [load, live]);

  const cams = cameras.length ? cameras : Object.keys(tl?.tracks_started ?? {}).sort();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, opacity: loading && tl ? 0.7 : 1 }}>
      <div className="controls">
        <div className="seg" role="group" aria-label="Time window">
          {RANGES.map(([label, s]) => <button key={s} aria-pressed={span === s} onClick={() => setSpan(s)}>{label}</button>)}
        </div>
        <span className="muted">{end ? `${live ? "last" : "until the end of the stored data,"} ${fmtDuration(span)}${live ? ", refreshed every 5 s" : ""}` : ""}</span>
        {error && <span className="error">{error}</span>}
      </div>
      {tl && <Density tl={tl} cameras={cams} setTip={setTip} />}
      {tracks && tl && <Swimlanes tracks={tracks} cameras={cams} since={tl.since} until={tl.until} setTip={setTip} onIdentity={onIdentity} />}
      {!tracks && tl && <div className="card empty">Choose a window of 15 minutes or less to see individual tracks.</div>}
      <Tooltip tip={tip} />
    </div>
  );
}
