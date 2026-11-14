import type { ReactNode } from "react";
import type { ModelInfo, Status, SystemInfo } from "../api";
import { compact, fmtDuration } from "../lib/charts";
import { IconChip, IconDatabase, IconGauge, IconLayers } from "./Icons";
import { Card, PageHead } from "./Layout";
import { StatusBadge } from "./Live";

function Kv({ k, v }: { k: string; v: ReactNode }) {
  return <div className="kv"><span>{k}</span><b>{v ?? "–"}</b></div>;
}

function Stage({ name, title, children }: { name: string; title: ReactNode; children: ReactNode }) {
  return (
    <div className="stage">
      <div className="stage-top">{name}</div>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

/** A meter: the fill carries the level (accent, then warning, then critical); the number is always printed. */
function Meter({ label, value, max, unit, note, severity = false }: {
  label: string; value: number | null | undefined; max: number; unit: string; note?: ReactNode; severity?: boolean;
}) {
  const frac = value == null ? 0 : Math.max(0, Math.min(1, value / max));
  // only where a high value is a problem (memory, lag); a busy GPU near its power limit is normal under load
  const level = !severity ? "" : frac >= 0.95 ? "critical" : frac >= 0.85 ? "warning" : "";
  return (
    <div className="card meter">
      <div className="meter-head"><span>{label}</span><b className="num">{value == null ? "–" : `${Math.round(value)}`}<small className="muted" style={{ fontSize: 12, fontWeight: 500 }}> {unit}</small></b></div>
      <div className="meter-track" role="meter" aria-label={label} aria-valuenow={value ?? undefined} aria-valuemin={0} aria-valuemax={max}>
        <div className={`meter-fill ${level}`} style={{ width: `${100 * frac}%` }} />
      </div>
      {note && <div className="meter-note">{note}</div>}
    </div>
  );
}

const model = (m: ModelInfo | null | undefined) => (m ? `${m.model_id ?? m.backend ?? "model"}` : "–");

export function SystemView({ status, system }: { status: Status | null; system: SystemInfo | null }) {
  const env = system?.environment;
  const gpuMem = Number((env?.gpu?.memory_total ?? "").split(" ")[0]) || 8192;
  const gpuPower = Number((env?.gpu?.power_limit ?? "").split(" ")[0]) || null;
  const reid = status?.reid;
  const dropPct = reid?.submitted ? (100 * (reid.dropped ?? 0)) / reid.submitted : null;
  const cams = Object.entries(status?.cameras ?? {}).sort(([a], [b]) => a.localeCompare(b));
  return (
    <>
      <PageHead title="System" sub={system ? `${env?.gpu?.name ?? "GPU"} · ${env?.cpu ?? ""}` : "Loading the system description…"} />

      <Card title="Pipeline" sub="what every processed frame goes through, as configured on this server" icon={<IconLayers size={16} />}>
        <div className="pipeline">
          <Stage name="Cameras" title={`${system?.cameras?.length ?? cams.length} sources`}>
            <Kv k="streams" v={system?.cameras?.filter((c) => c.kind === "stream").length ?? "–"} />
            <Kv k="video files" v={system?.cameras?.filter((c) => c.kind === "file").length ?? "–"} />
            <Kv k="decode" v={`FFmpeg, OpenCV ${system?.opencv_threads ?? "default"} thr.`} />
          </Stage>
          <Stage name="Detect" title={model(system?.detector)}>
            <Kv k="precision" v={system?.detector?.dtype} />
            <Kv k="input" v={system?.detector?.imgsz ? `${system.detector.imgsz} px` : "–"} />
            <Kv k="batch" v={system?.pipeline ? `up to ${system.pipeline.batch_size} cameras` : "–"} />
            <Kv k="rate" v={system?.pipeline ? (system.pipeline.process_fps ? `${system.pipeline.process_fps} fps / camera` : "every frame") : "–"} />
          </Stage>
          <Stage name="Track" title={system?.pipeline?.tracker ?? "ByteTrack"}>
            <Kv k="classes" v={system?.pipeline?.phrases.join(", ")} />
            <Kv k="threshold" v={system?.pipeline?.track_thresh} />
            <Kv k="grouped" v={Object.keys(system?.pipeline?.phrase_groups ?? {}).join(", ") || "none"} />
          </Stage>
          <Stage name="Re-identify" title={system?.reid ? model(system.reid.embedder) : "off"}>
            <Kv k="precision" v={system?.reid?.embedder?.dtype} />
            <Kv k="thread" v={system?.reid ? (system.reid.asynchronous ? "own worker" : "detection worker") : "–"} />
            <Kv k="link threshold" v={system?.reid?.appearance_threshold} />
          </Stage>
          <Stage name="Store & search" title="SQLite + CLIP">
            <Kv k="retention" v={system?.store.retention_hours != null ? `${system.store.retention_hours} h` : "unlimited"} />
            <Kv k="thumbnails" v={system ? (system.store.thumbnails ? "kept" : "off") : "–"} />
            <Kv k="type check" v={system?.search ? (system.search.vehicle_type_check ? "on" : "off") : "–"} />
          </Stage>
        </div>
      </Card>

      <div className="meters">
        <Meter label="GPU utilisation" value={status?.gpu?.util_percent} max={100} unit="%"
               note="share of time a GPU kernel was running (nvidia-smi)" />
        <Meter label="GPU memory" severity value={status?.gpu?.memory_used_mib} max={gpuMem} unit="MiB" note={`of ${compact(gpuMem)} MiB`} />
        <Meter label="GPU power" value={status?.gpu?.power_w ?? null} max={gpuPower ?? 60} unit="W"
               note={gpuPower ? `limit ${gpuPower} W` : "board power"} />
        <Meter label="Process CPU" value={status?.process_cpu_percent} max={100 * (env?.cpu_logical_cores ?? 16)} unit="%"
               note={`100 % = one core; ${env?.cpu_logical_cores ?? "–"} logical cores`} />
        <Meter label="Re-ID lag" severity value={reid?.lag_ms_p50 ?? null} max={1000} unit="ms"
               note={dropPct == null ? "Re-ID runs in the detection worker" : `median; ${dropPct.toFixed(1)} % of frames skipped by Re-ID`} />
        <Meter label="Database" value={status?.store.size_mb} max={Math.max(1024, (status?.store.size_mb ?? 0) * 1.25)} unit="MB"
               note={`${compact(status?.store.written ?? 0)} writes · ${status?.store.dropped ?? 0} dropped`} />
      </div>

      <Card title="Cameras" sub="health and per-camera processing, refreshed every second" icon={<IconGauge size={16} />} bodyClass="card-b table-wrap">
        {cams.length === 0 ? <div className="empty-state" style={{ padding: 20 }}>No live cameras.</div> : (
          <table className="data">
            <thead><tr><th>camera</th><th>state</th><th className="n">decoded fps</th><th className="n">processed fps</th>
              <th className="n">latency p50 / p90</th><th className="n">in view</th><th className="n">reconnects</th><th>source</th></tr></thead>
            <tbody>
              {cams.map(([id, c]) => (
                <tr key={id}>
                  <td><b>{id}</b></td><td><StatusBadge camera={c} /></td>
                  <td className="n">{c.input_fps?.toFixed(1) ?? "–"}</td><td className="n">{c.processed_fps?.toFixed(1) ?? "–"}</td>
                  <td className="n">{c.latency_ms_p50 != null ? `${Math.round(c.latency_ms_p50)} / ${Math.round(c.latency_ms_p90 ?? 0)} ms` : "–"}</td>
                  <td className="n">{c.active_tracks ?? 0}</td><td className="n">{Math.max(0, (c.connects ?? 1) - 1)}</td>
                  <td className="mono muted">{c.url}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <div className="meters two">
        <Card title="Hardware and software" icon={<IconChip size={16} />}>
          <div className="kv-grid">
            <Kv k="GPU" v={env?.gpu?.name} /><Kv k="driver" v={env?.gpu?.driver} />
            <Kv k="CUDA (PyTorch build)" v={env?.torch_cuda_build} /><Kv k="PyTorch" v={env?.torch} />
            <Kv k="CPU" v={env?.cpu} /><Kv k="RAM" v={env?.ram_gib ? `${env.ram_gib} GiB` : null} />
            <Kv k="Python" v={env?.python} /><Kv k="OS" v={env?.os?.split("-with")[0]} />
          </div>
        </Card>
        <Card title="Store and stream" icon={<IconDatabase size={16} />}>
          <div className="kv-grid">
            <Kv k="run" v={status?.run_id ?? "replay"} /><Kv k="uptime" v={status ? fmtDuration(status.uptime_s) : null} />
            <Kv k="tracks in search index" v={compact(status?.index_tracks ?? 0)} />
            <Kv k="last commit" v={status ? `${status.store.last_commit_ms.toFixed(1)} ms` : null} />
            <Kv k="live frames encoded" v={compact(status?.live_hub?.encodes ?? 0)} />
            <Kv k="JPEG encode, mean" v={status?.live_hub ? `${status.live_hub.encode_ms_mean.toFixed(1)} ms` : null} />
            <Kv k="trajectory points" v={system ? `every ${system.store.points_every_s} s` : null} />
            <Kv k="search weights" v={system?.search ? `text ${system.search.weights.text} · attributes ${system.search.weights.attributes}` : null} />
          </div>
        </Card>
      </div>
    </>
  );
}
