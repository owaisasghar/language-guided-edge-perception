import { useEffect, useState } from "react";
import { getJSON, thumbUrl, type Identity } from "../api";
import { fmtClock, fmtDuration } from "../lib/charts";

/** "Which cameras saw it?": every track of one global identity, in time order, with the link evidence. */
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

  return (
    <div className="drawer-bg" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Identity" onClick={(e) => e.stopPropagation()}>
        <div className="row">
          <h2 className="grow">{d ? `Identity G${d.global_id}` : "Identity"}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">Close</button>
        </div>
        {error && <div className="error">{error}</div>}
        {!d && !error && <div className="muted">Loading…</div>}
        {d && (
          <>
            <div className="secondary">
              Seen by {d.cameras.length} camera{d.cameras.length === 1 ? "" : "s"} ({d.cameras.join(", ")}) in {d.n_tracks} track
              {d.n_tracks === 1 ? "" : "s"}, {fmtClock(d.t_start)} – {fmtClock(d.t_end)} (run {d.run_id}).
              Global IDs are assigned when a track ends, by appearance and time; each link below shows its evidence.
            </div>
            <div>
              {d.tracks?.map((t) => (
                <div className="track-row" key={t.id}>
                  {t.has_thumbnail ? <img src={thumbUrl(t.id)} alt="" loading="lazy" /> : <div style={{ width: 40 }} />}
                  <div className="grow">
                    <div><b>{t.camera_id}</b> · track {t.local_id} · {t.category}</div>
                    <div className="muted num">{fmtClock(t.t_start)} – {fmtClock(t.t_end)} ({fmtDuration(t.duration_s)})
                      {t.decision ? ` · ${t.decision.replace(/_/g, " ")}` : ""}{t.link_score != null ? ` (${t.link_score.toFixed(2)})` : ""}</div>
                  </div>
                </div>
              ))}
            </div>
            {!!d.links?.length && (
              <div>
                <h3 style={{ fontSize: 14, margin: "8px 0" }}>Cross-camera links</h3>
                <table className="data">
                  <thead><tr><th>time</th><th>camera</th><th>after</th><th className="n">similarity</th></tr></thead>
                  <tbody>
                    {d.links.map((l) => {
                      const x = l.data as { camera?: string; from_cameras?: string[]; score?: number };
                      return (
                        <tr key={l.id}>
                          <td className="num">{fmtClock(l.t)}</td><td>{x.camera}</td><td>{(x.from_cameras ?? []).join(", ")}</td>
                          <td className="n">{x.score?.toFixed(3) ?? "–"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </aside>
    </div>
  );
}
