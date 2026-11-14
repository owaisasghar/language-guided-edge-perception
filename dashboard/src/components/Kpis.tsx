import type { ReactNode } from "react";
import type { Status } from "../api";
import type { Sample } from "../hooks";
import { compact, sparkPath } from "../lib/charts";
import { IconCamera, IconClock, IconGauge, IconLayers, IconLink, IconSpark } from "./Icons";

/** A single-series trend: line in a neutral ink, the latest point in the accent colour; no axes (the value
 *  next to it carries the number). */
export function Sparkline({ values, w = 96, h = 30, max, label }: {
  values: (number | null)[]; w?: number; h?: number; max?: number; label: string;
}) {
  const d = sparkPath(values, w, h, 3, max);
  const last = [...values].reverse().find((v) => v != null);
  const lastIdx = values.length - 1 - [...values].reverse().findIndex((v) => v != null);
  if (!d || last == null) return <svg className="spark" width={w} height={h} role="img" aria-label={`${label}: no data yet`} />;
  const top = Math.max(max ?? 0, ...values.filter((v): v is number => v != null), 1e-9);
  const cx = 3 + (lastIdx / (values.length - 1)) * (w - 6);
  const cy = h - 3 - (last / top) * (h - 6);
  return (
    <svg className="spark" width={w} height={h} role="img" aria-label={`${label}, last two minutes`}>
      <path d={d} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={cx} cy={cy} r={4} fill="var(--accent)" stroke="var(--surface-1)" strokeWidth={2} />
    </svg>
  );
}

function Kpi({ icon, label, value, unit, note, spark }: {
  icon: ReactNode; label: string; value: string; unit?: string; note?: ReactNode; spark?: ReactNode;
}) {
  return (
    <div className="card kpi">
      <div className="kpi-top">{icon}<span>{label}</span></div>
      <div className="kpi-mid">
        <div className="kpi-value num">{value}{unit && <small>{unit}</small>}</div>
        {spark}
      </div>
      {note && <div className="kpi-note">{note}</div>}
    </div>
  );
}

export function KpiStrip({ status, history, closedTracks }: { status: Status | null; history: Sample[]; closedTracks: number }) {
  const cams = Object.values(status?.cameras ?? {});
  const live = cams.filter((c) => c.state === "streaming" && !c.stalled).length;
  const fps = cams.reduce((s, c) => s + (c.processed_fps ?? 0), 0);
  const lat = cams.map((c) => c.latency_ms_p50).filter((v): v is number => v != null).sort((a, b) => a - b);
  const target = status?.pipeline?.target_process_fps;
  const gpu = status?.gpu;
  return (
    <section className="kpis" aria-label="System overview">
      <Kpi icon={<IconCamera size={16} />} label="Cameras online" value={status ? `${live}` : "–"}
           unit={status ? ` / ${cams.length}` : undefined}
           note={status?.live === false ? "replay of stored runs" : live === cams.length ? "all streaming" : `${cams.length - live} not streaming`} />
      <Kpi icon={<IconGauge size={16} />} label="Throughput" value={status ? fps.toFixed(1) : "–"} unit=" fps"
           note={target ? `target ${target} fps per camera` : "frames processed per second"}
           spark={<Sparkline values={history.map((s) => s.fps)} w={72} label="Throughput" />} />
      <Kpi icon={<IconClock size={16} />} label="Latency (median)"
           value={lat.length ? `${Math.round(lat[Math.floor(lat.length / 2)])}` : "–"} unit={lat.length ? " ms" : undefined}
           note="frame decoded → tracks ready" />
      <Kpi icon={<IconSpark size={16} />} label="GPU" value={gpu ? `${Math.round(gpu.util_percent)}` : "–"} unit={gpu ? " %" : undefined}
           note={gpu ? `${compact(gpu.memory_used_mib)} MiB${gpu.power_w ? ` · ${gpu.power_w.toFixed(0)} W` : ""}` : "no reading"}
           spark={<Sparkline values={history.map((s) => s.gpu)} w={72} max={100} label="GPU utilisation" />} />
      <Kpi icon={<IconLayers size={16} />} label="Tracks (last hour)" value={compact(status?.last_hour?.tracks ?? 0)}
           note={`${closedTracks} finished since page load`} />
      <Kpi icon={<IconLink size={16} />} label="Cross-camera IDs" value={compact(status?.last_hour?.multi_camera_identities ?? 0)}
           note={`${compact(status?.last_hour?.identity_links ?? 0)} links in the last hour`} />
    </section>
  );
}
