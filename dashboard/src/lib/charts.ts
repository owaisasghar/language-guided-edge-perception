// Pure helpers for the charts and status displays (unit-tested in charts.test.ts).

export interface Interval {
  t_start: number;
  t_end: number;
}

/** Greedy interval packing: each item gets the lowest row whose previous item ended (plus `gap`) before it starts.
 *  Items are processed by start time; returns row per item (same order as input) and the row count. */
export function packRows<T extends Interval>(items: T[], gap = 0): { rows: number[]; count: number } {
  const order = items.map((_, i) => i).sort((a, b) => items[a].t_start - items[b].t_start || a - b);
  const ends: number[] = [];
  const rows = new Array<number>(items.length);
  for (const i of order) {
    let r = ends.findIndex((e) => e + gap <= items[i].t_start);
    if (r < 0) {
      r = ends.length;
      ends.push(0);
    }
    ends[r] = items[i].t_end;
    rows[i] = r;
  }
  return { rows, count: ends.length };
}

const STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];

/** A "nice" step (seconds) so that [since, until] has about `target` steps. */
export function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  return STEPS_S.find((s) => s >= raw) ?? STEPS_S[STEPS_S.length - 1];
}

/** Tick times aligned to the step (in local time for steps of a minute or more). */
export function timeTicks(since: number, until: number, target = 6): number[] {
  const step = niceStep(until - since, target);
  const offset = step >= 60 ? new Date(since * 1000).getTimezoneOffset() * 60 : 0; // align to local clock
  const first = Math.ceil((since - offset) / step) * step + offset;
  const out: number[] = [];
  for (let t = first; t <= until + 1e-9; t += step) out.push(t);
  return out;
}

/** Heatmap bucket size: a nice step giving at most `maxBuckets` buckets. */
export function bucketSeconds(span: number, maxBuckets = 60): number {
  return STEPS_S.find((s) => span / s <= maxBuckets) ?? STEPS_S[STEPS_S.length - 1];
}

/** Count -> sequential step 1..steps (0 = empty), on a square-root scale so small counts stay visible. */
export function seqStep(count: number, max: number, steps = 7): number {
  if (count <= 0 || max <= 0) return 0;
  return Math.min(steps, Math.max(1, Math.ceil(Math.sqrt(count / max) * steps)));
}

export type Level = "good" | "warning" | "serious" | "critical";

/** Camera health as status level + icon + label (status colours never carry meaning alone). */
export function cameraHealth(c: { state: string; stalled?: boolean } | undefined): { level: Level; icon: string; label: string } {
  if (!c) return { level: "critical", icon: "×", label: "Unknown" };
  if (c.state === "streaming" && c.stalled) return { level: "serious", icon: "!", label: "Stalled" };
  switch (c.state) {
    case "streaming":
      return { level: "good", icon: "✓", label: "Live" };
    case "connecting":
      return { level: "warning", icon: "…", label: "Connecting" };
    case "reconnecting":
      return { level: "serious", icon: "↻", label: "Reconnecting" };
    default:
      return { level: "critical", icon: "×", label: c.state.charAt(0).toUpperCase() + c.state.slice(1) };
  }
}

export function fmtClock(t: number, seconds = true): string {
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: seconds ? "2-digit" : undefined });
}

export function fmtDuration(s: number): string {
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

export function fmtAgo(t: number, now: number = Date.now() / 1000): string {
  const d = Math.max(0, now - t);
  if (d < 60) return `${Math.round(d)} s ago`;
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  return `${Math.round(d / 3600)} h ago`;
}

/** 1,284 / 12.9K / 4.2M */
export function compact(n: number): string {
  if (Math.abs(n) < 10_000) return Math.round(n).toLocaleString("en-US");
  if (Math.abs(n) < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}
