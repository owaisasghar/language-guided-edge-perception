import { useEffect, useRef, useState } from "react";
import { ApiError, postJSON, thumbUrl, type Health, type Parsed, type SearchHit, type SearchResponse } from "../api";
import { useNow } from "../hooks";
import { compact, fmtAgo, fmtClock, fmtDuration, SWATCH } from "../lib/charts";
import { IconAlert, IconCamera, IconCar, IconClock, IconLink, IconSearch, IconUser } from "./Icons";
import { PageHead } from "./Layout";

const PEOPLE = ["person in a white shirt", "woman in red", "person with a backpack", "man in a black jacket on cam01",
                "person in blue in the last 5 minutes"];

function Swatch({ colour }: { colour: string }) {
  return <span className="swatch" style={{ background: SWATCH[colour] ?? "transparent" }} aria-hidden="true" />;
}

/** How the query was understood, as labelled chips (the words carry the meaning; swatches only illustrate). */
function Understood({ p }: { p: Parsed }) {
  const cats = p.categories.length ? p.categories.join(" or ") : "anything";
  return (
    <div className="understood" aria-label="How the query was understood">
      <span className="muted" style={{ fontSize: 12.5 }}>Understood as</span>
      <span className="chip"><span className="k">object</span><b>{cats}</b></span>
      {p.object_colours.map((c) => <span className="chip" key={`c${c}`}><Swatch colour={c} /><span className="k">colour</span><b>{c}</b></span>)}
      {(p.vehicle_types ?? []).map((t) => <span className="chip" key={`t${t}`}><IconCar size={13} /><span className="k">type</span><b>{t}</b></span>)}
      {p.clothing.map((c, i) => (
        <span className="chip" key={`w${i}`}><Swatch colour={c.colour} /><span className="k">{c.item ?? `${c.region} clothing`}</span><b>{c.colour}</b></span>
      ))}
      {p.carried.map((c) => <span className="chip" key={`r${c}`}><span className="k">with</span><b>{c}</b></span>)}
      {p.cameras.length > 0 && <span className="chip"><IconCamera size={13} /><span className="k">camera</span><b>{p.cameras.join(", ")}</b></span>}
      {p.time?.last_s && <span className="chip"><IconClock size={13} /><span className="k">time</span><b>last {fmtDuration(p.time.last_s)}</b></span>}
      {p.time?.after && <span className="chip"><IconClock size={13} /><span className="k">after</span><b>{p.time.after}</b></span>}
      {p.time?.before && <span className="chip"><IconClock size={13} /><span className="k">before</span><b>{p.time.before}</b></span>}
    </div>
  );
}

/** Weighted contributions to the ranking score (description similarity, attribute checks): a two-part bar. */
function ScoreBar({ hit, weights, max }: { hit: SearchHit; weights: SearchResponse["weights"]; max: number }) {
  const t = Math.max(0, weights.text * hit.text);
  const a = Math.max(0, weights.attributes * hit.attributes);
  const pct = (v: number) => `${(100 * v) / Math.max(max, 1e-9)}%`;
  return (
    <div title={`description ${hit.text.toFixed(3)} × ${weights.text} + attributes ${hit.attributes.toFixed(2)} × ${weights.attributes}`}>
      <div className="scorebar" role="img" aria-label={`score ${hit.score.toFixed(2)}: description ${t.toFixed(2)}, attributes ${a.toFixed(2)}`}>
        <span style={{ width: pct(t), background: "var(--series-1)" }} />
        {a > 0 && <span style={{ width: pct(a), background: "var(--series-2)" }} />}
      </div>
      <div className="score-row num"><span>description {hit.text.toFixed(2)}</span><span>attributes {hit.attributes.toFixed(2)}</span></div>
    </div>
  );
}

function HitCard({ h, weights, max, now, onIdentity }: {
  h: SearchHit; weights: SearchResponse["weights"]; max: number; now: number; onIdentity: (id: number) => void;
}) {
  const vehicle = h.category !== "person";
  return (
    <button className="hit" disabled={!h.identity_id} onClick={() => h.identity_id && onIdentity(h.identity_id)}
            aria-label={`Result ${h.rank}: ${h.category} on ${h.camera}`}>
      <div className="hit-media">
        {h.track_id && h.has_thumbnail ? <img src={thumbUrl(h.track_id)} alt="" loading="lazy" />
          : <span className="noimg">{vehicle ? <IconCar size={30} /> : <IconUser size={30} />}</span>}
        <span className="rank">{h.rank}</span>
        {h.global_id != null && <span className="gid">G{h.global_id}</span>}
      </div>
      <div className="hit-body">
        <div className="hit-title">{h.category}<span className="cam-chip">{h.camera}</span></div>
        <div className="hit-meta num"><IconClock size={13} />{fmtClock(h.t_start)} · {fmtDuration(h.duration_s)} · {fmtAgo(h.t_end, now)}</div>
        <div className="hit-meta"><IconLink size={13} />seen in {h.seen_in_cameras.map((c) => <span className="cam-chip" key={c}>{c}</span>)}</div>
        <ScoreBar hit={h} weights={weights} max={max} />
      </div>
    </button>
  );
}

const TIPS: [string, string][] = [
  ["Colours", "\"woman in red\", \"white SUV\", \"black jacket\""],
  ["Clothing and regions", "\"red shirt\" checks the upper body, \"blue jeans\" the lower body"],
  ["Vehicle types", "sedan, SUV, pickup, van, minivan, hatchback, wagon"],
  ["Cameras and time", "\"on cam02\", \"in the last 10 minutes\", \"after 14:30\""],
];

export function SearchView({ health, onIdentity }: { health: Health | null; onIdentity: (id: number) => void }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const now = useNow(10000);
  const examples = health?.query_examples?.length ? health.query_examples : PEOPLE;

  async function run(query: string) {
    if (!query.trim()) return;
    setQ(query);
    setBusy(true);
    setError(null);
    try {
      setRes(await postJSON<SearchResponse>("/api/search", { query, k: 24 }));
    } catch (e) {
      setError(e instanceof ApiError ? (e.status === 503 ? "Search is not available on this server (no CLIP model loaded)." : `${e.status}: ${e.message}`) : String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { // a link like /?q=woman+in+red#search runs its query
    const q0 = new URLSearchParams(window.location.search).get("q");
    if (q0) run(q0);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const max = res ? Math.max(...res.hits.map((h) => Math.max(0, res.weights.text * h.text) + Math.max(0, res.weights.attributes * h.attributes)), 0) : 0;
  return (
    <>
      <PageHead title="Search" sub="Find people and vehicles across every camera by describing them in plain language." />
      <section className="card hero">
        <div>
          <h2>What are you looking for?</h2>
          <p>Each finished track is searchable: its colours, clothing, vehicle type and appearance are compared with your words.</p>
        </div>
        <form className="searchbox" role="search" onSubmit={(e) => { e.preventDefault(); run(q); }}>
          <IconSearch size={20} />
          <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} maxLength={500}
                 placeholder={`Try "${examples[0]}"`} aria-label="Describe what to find" />
          <kbd aria-hidden="true">/</kbd>
          <button className="btn" disabled={busy || !q.trim()}>{busy ? "Searching…" : "Search"}</button>
        </form>
        <div className="row">
          <span className="muted" style={{ fontSize: 12.5 }}>Examples</span>
          {examples.map((x) => <button type="button" className="chip" key={x} onClick={() => run(x)}>{x}</button>)}
        </div>
      </section>

      {error && <div className="banner error" role="alert"><IconAlert size={16} />{error}</div>}

      {busy && !res && (
        <div className="results">{Array.from({ length: 8 }, (_, i) => <div key={i} className="skeleton" style={{ height: 300 }} />)}</div>
      )}

      {!res && !busy && (
        <section className="card empty-state">
          <div className="art"><IconSearch size={28} /></div>
          <h3>Search everything the cameras have seen</h3>
          <div>Results are grouped by identity: one card per person or vehicle, with the cameras where it appeared.</div>
          <div className="tips">{TIPS.map(([t, d]) => <div className="tip" key={t}><b>{t}</b>{d}</div>)}</div>
        </section>
      )}

      {res && (
        <div className="stack" style={{ opacity: busy ? 0.6 : 1, transition: "opacity 0.15s" }}>
          <Understood p={res.parsed} />
          <div className="results-head">
            <span className="secondary"><b className="num" style={{ color: "var(--text-primary)" }}>{res.hits.length}</b> results from{" "}
              <span className="num">{compact(res.indexed_tracks)}</span> finished tracks · <span className="num">{res.timing_ms.total.toFixed(0)} ms</span></span>
            <span className="grow" />
            <span className="legend" aria-label="Score parts">
              <span><span className="key" style={{ background: "var(--series-1)" }} />description similarity</span>
              <span><span className="key" style={{ background: "var(--series-2)" }} />colour / type check</span>
            </span>
          </div>
          {res.hits.length === 0 ? (
            <section className="card empty-state">
              <div className="art"><IconSearch size={28} /></div>
              <h3>Nothing matches yet</h3>
              <div>Tracks become searchable about three seconds after they leave a camera's view. Try a broader description.</div>
            </section>
          ) : (
            <div className="results">
              {res.hits.map((h) => <HitCard key={`${h.run_id}-${h.track}`} h={h} weights={res.weights} max={max} now={now} onIdentity={onIdentity} />)}
            </div>
          )}
        </div>
      )}
    </>
  );
}
