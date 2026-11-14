import { useEffect, useState } from "react";
import { getJSON, thumbUrl, type Identity, type Track } from "../api";
import { fmtClock, fmtDuration, fmtGap } from "../lib/charts";
import { IconArrowRight, IconCamera, IconClock, IconLink, IconUser, IconX } from "./Icons";

function decisionText(t: Track): string {
  if (!t.decision || t.decision === "new" || t.decision === "new_low_evidence") return "first sighting";
  if (t.decision === "matched") return `linked${t.link_score != null ? ` · similarity ${t.link_score.toFixed(2)}` : ""}`;
  return t.decision.replace(/_/g, " ");
}

/** "Which cameras saw it?": every appearance of one global identity in time order, with the gaps between them. */
export function IdentityDrawer({ id, onClose }: { id: number; onClose: () => void }) {
  const [d, setD] = useState<Identity | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    setD(null);
    getJSON<Identity>(`/api/identities/${id}`, {}, ctrl.signal).then(setD).catch((e) => !ctrl.signal.aborted && setError(String(e)));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      ctrl.abort();
      window.removeEventListener("keydown", onKey);
    };
  }, [id, onClose]);

  const tracks = d?.tracks ?? [];
  const cover = tracks.find((t) => t.has_thumbnail);
  return (
    <div className="drawer-bg" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Identity details" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          {cover ? <img src={thumbUrl(cover.id)} alt="" /> : <div className="ph" />}
          <div className="grow">
            <h2>{d ? `Identity G${d.global_id}` : "Identity"}</h2>
            {d && <div className="row" style={{ marginTop: 6 }}>
              {d.cameras.map((c) => <span className="cam-chip" key={c}>{c}</span>)}
              <span className="muted" style={{ fontSize: 12 }}>run {d.run_id}</span>
            </div>}
          </div>
          <button className="btn-ghost" onClick={onClose} aria-label="Close"><IconX size={16} /></button>
        </div>
        <div className="drawer-body">
          {error && <div className="banner error">{error}</div>}
          {!d && !error && <div className="skeleton" style={{ height: 220 }} />}
          {d && (
            <>
              <div className="facts">
                <div className="fact"><div className="v num">{d.cameras.length}</div><div className="l">camera{d.cameras.length === 1 ? "" : "s"}</div></div>
                <div className="fact"><div className="v num">{d.n_tracks}</div><div className="l">appearances</div></div>
                <div className="fact"><div className="v num">{fmtDuration(Math.max(0, d.t_end - d.t_start))}</div><div className="l">first to last sighting</div></div>
              </div>
              <section>
                <div className="row" style={{ marginBottom: 4 }}><IconArrowRight size={15} /><b>Journey</b>
                  <span className="muted" style={{ fontSize: 12 }}>{fmtClock(d.t_start)} – {fmtClock(d.t_end)}</span></div>
                <ol className="journey">
                  {tracks.map((t, i) => (
                    <li key={t.id}>
                      {i > 0 && <div className="transit" style={{ marginLeft: 38 }}><IconClock size={12} />{fmtGap(tracks[i - 1].t_end, t.t_start)}
                        {tracks[i - 1].camera_id !== t.camera_id ? ` · ${tracks[i - 1].camera_id} → ${t.camera_id}` : " · same camera"}</div>}
                      <div className="step">
                        <div className="rail"><span className="node" /><span className="line" /></div>
                        <div className="step-card">
                          {t.has_thumbnail ? <img src={thumbUrl(t.id)} alt="" loading="lazy" /> : <div className="ph" />}
                          <div className="grow">
                            <div className="row"><span className="cam-chip">{t.camera_id}</span><b>track {t.local_id}</b>
                              <span className="secondary">{t.category}</span></div>
                            <div className="muted num" style={{ fontSize: 12, marginTop: 4 }}>
                              {fmtClock(t.t_start)} – {fmtClock(t.t_end)} · {fmtDuration(t.duration_s)}</div>
                            <div className="secondary" style={{ fontSize: 12, marginTop: 2 }}>{decisionText(t)}</div>
                          </div>
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              </section>
              {!!d.links?.length && (
                <section className="card">
                  <div className="card-h"><span className="card-icon"><IconLink size={16} /></span>
                    <div className="grow"><h2>Cross-camera links</h2><div className="sub">appearance similarity when the identity reached a new camera</div></div></div>
                  <div className="card-b table-wrap">
                    <table className="data">
                      <thead><tr><th>time</th><th>camera</th><th>after</th><th className="n">similarity</th></tr></thead>
                      <tbody>
                        {d.links.map((l) => {
                          const x = l.data as { camera?: string; from_cameras?: string[]; score?: number };
                          return (
                            <tr key={l.id}><td className="num">{fmtClock(l.t)}</td><td>{x.camera}</td>
                              <td>{(x.from_cameras ?? []).join(", ")}</td><td className="n">{x.score?.toFixed(3) ?? "–"}</td></tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
              <p className="muted" style={{ fontSize: 12, margin: 0, display: "flex", gap: 8 }}>
                <IconUser size={14} /><span>A global identity is assigned when a track ends, from appearance (CLIP) and timing between cameras.
                  Links can be wrong when two people or vehicles look alike; each link shows its evidence.</span>
              </p>
              <p className="muted" style={{ fontSize: 12, margin: 0, display: "flex", gap: 8 }}>
                <IconCamera size={14} /><span>Thumbnails are the best crop of each track, kept for the configured retention period.</span>
              </p>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
