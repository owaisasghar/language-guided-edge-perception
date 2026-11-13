import { useEffect, useState } from "react";
import { ApiError, postJSON, thumbUrl, type Parsed, type SearchHit, type SearchResponse, type Status } from "../api";
import { compact, fmtClock, fmtDuration } from "../lib/charts";

export function StatTiles({ status, closedTracks, connected }: { status: Status | null; closedTracks: number; connected: boolean }) {
  const cams = Object.values(status?.cameras ?? {});
  const live = cams.filter((c) => c.state === "streaming" && !c.stalled).length;
  const fps = cams.reduce((s, c) => s + (c.processed_fps ?? 0), 0);
  const lat = cams.map((c) => c.latency_ms_p50).filter((v): v is number => v != null).sort((a, b) => a - b);
  const tiles: { label: string; value: string; note?: string }[] = [
    { label: "Cameras live", value: !status ? "–" : status.live ? `${live} / ${cams.length}` : "replay",
      note: connected ? "stream connected" : "stream reconnecting…" },
    { label: "Processed frames per second", value: status?.live ? fps.toFixed(1) : "–", note: status?.pipeline ? `target ${status.pipeline.target_process_fps} per camera` : undefined },
    { label: "Latency, median camera", value: lat.length ? `${Math.round(lat[Math.floor(lat.length / 2)])} ms` : "–", note: "frame decoded → tracks ready" },
    { label: "GPU", value: status?.gpu ? `${Math.round(status.gpu.util_percent)} %` : "–", note: status?.gpu ? `${compact(status.gpu.memory_used_mib)} MiB` : undefined },
    { label: "CPU (this process)", value: status?.process_cpu_percent != null ? `${Math.round(status.process_cpu_percent)} %` : "–", note: "100 % = one core" },
    { label: "Tracks, last hour", value: compact(status?.last_hour?.tracks ?? 0), note: `${closedTracks} finished since page load` },
    { label: "Cross-camera identities", value: compact(status?.last_hour?.multi_camera_identities ?? 0), note: "last hour" },
  ];
  return (
    <div className="tiles">
      {tiles.map((t) => (
        <div className="card tile" key={t.label}>
          <div className="label">{t.label}</div>
          <div className="value">{t.value}</div>
          {t.note && <div className="note">{t.note}</div>}
        </div>
      ))}
    </div>
  );
}

function Chips({ p }: { p: Parsed }) {
  const chips: [string, string][] = [];
  chips.push(["object", p.categories.length ? p.categories.join(" or ") : "any"]);
  p.object_colours.forEach((c) => chips.push(["colour", c]));
  p.clothing.forEach((c) => chips.push([c.item ?? `${c.region} clothing`, c.colour]));
  p.carried.forEach((c) => chips.push(["with", c]));
  if (p.cameras.length) chips.push(["camera", p.cameras.join(", ")]);
  if (p.time?.last_s) chips.push(["time", `last ${fmtDuration(p.time.last_s)}`]);
  if (p.time?.after) chips.push(["after", p.time.after]);
  if (p.time?.before) chips.push(["before", p.time.before]);
  return (
    <div className="row" aria-label="How the query was understood">
      <span className="muted">Understood as</span>
      {chips.map(([k, v], i) => <span className="chip" key={i}>{k} <b>{v}</b></span>)}
    </div>
  );
}

/** Weighted contributions to the ranking score (description similarity, colour check): a two-part bar. */
function ScoreBar({ hit, weights, max }: { hit: SearchHit; weights: SearchResponse["weights"]; max: number }) {
  const t = Math.max(0, weights.text * hit.text);
  const a = Math.max(0, weights.attributes * hit.attributes);
  const pct = (v: number) => `${(100 * v) / Math.max(max, 1e-9)}%`;
  return (
    <div title={`description ${hit.text.toFixed(3)} × ${weights.text} + colour ${hit.attributes.toFixed(2)} × ${weights.attributes}`}>
      <div className="scorebar" role="img" aria-label={`score ${hit.score.toFixed(2)}: description ${t.toFixed(2)}, colour ${a.toFixed(2)}`}>
        <span style={{ width: pct(t), background: "var(--series-1)" }} />
        {a > 0 && <span style={{ width: pct(a), background: "var(--series-2)" }} />}
      </div>
      <div className="muted num">description {hit.text.toFixed(2)} · colour {hit.attributes.toFixed(2)}</div>
    </div>
  );
}

const EXAMPLES = ["person in a white shirt", "woman in red", "person with a backpack", "man in a black jacket on cam01",
                  "person in blue in the last 5 minutes"];

export function SearchView({ onIdentity }: { onIdentity: (id: number) => void }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(query: string) {
    if (!query.trim()) return;
    setQ(query);
    setBusy(true);
    setError(null);
    try {
      setRes(await postJSON<SearchResponse>("/api/search", { query, k: 24 }));
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.message}` : String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { // a link like /?q=woman+in+red#search runs its query
    const q0 = new URLSearchParams(window.location.search).get("q");
    if (q0) run(q0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const max = res ? Math.max(...res.hits.map((h) => Math.max(0, res.weights.text * h.text) + Math.max(0, res.weights.attributes * h.attributes)), 0) : 0;
  return (
    <div className="card">
      <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <form className="searchbar" onSubmit={(e) => { e.preventDefault(); run(q); }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder='Describe what to find, e.g. "person in a red shirt on cam02"'
                 aria-label="Search query" maxLength={500} />
          <button className="btn" disabled={busy || !q.trim()}>{busy ? "Searching…" : "Search"}</button>
        </form>
        <div className="row">
          <span className="muted">Try</span>
          {EXAMPLES.map((x) => <button type="button" className="chip" key={x} onClick={() => run(x)}>{x}</button>)}
        </div>
        {error && <div className="error">{error}</div>}
        {res && (
          <>
            <Chips p={res.parsed} />
            <div className="row">
              <span className="secondary">{res.hits.length} results from {compact(res.indexed_tracks)} finished tracks
                · {res.timing_ms.total.toFixed(0)} ms</span>
              <span className="grow" />
              <span className="legend" aria-label="Score parts">
                <span><span className="key" style={{ background: "var(--series-1)" }} />description similarity</span>
                <span><span className="key" style={{ background: "var(--series-2)" }} />colour check</span>
              </span>
            </div>
            {res.hits.length === 0 ? <div className="empty">Nothing matches. Tracks become searchable when they end.</div> : (
              <div className="results">
                {res.hits.map((h) => (
                  <button className="hit" key={`${h.run_id}-${h.track}`} disabled={!h.identity_id}
                          onClick={() => h.identity_id && onIdentity(h.identity_id)}>
                    {h.track_id && h.has_thumbnail ? <img className="thumb" src={thumbUrl(h.track_id)} alt="" loading="lazy" />
                                                   : <div className="thumb" />}
                    <div className="info">
                      <span className="title">#{h.rank} · {h.camera} · {h.category}</span>
                      <span className="secondary">{fmtClock(h.t_start)} · {fmtDuration(h.duration_s)}</span>
                      <span className="secondary">{h.global_id != null ? `G${h.global_id} · ` : ""}seen in {h.seen_in_cameras.join(", ")}</span>
                      <ScoreBar hit={h} weights={res.weights} max={max} />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
