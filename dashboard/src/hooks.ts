import { useEffect, useRef, useState } from "react";
import { apiUrl, getJSON, type EventItem, type Health, type Status, type SystemInfo } from "./api";

/** One point of the status history (sparklines): time, total processed fps, GPU %, process CPU %. */
export interface Sample {
  t: number;
  fps: number;
  gpu: number | null;
  cpu: number | null;
}

export interface LiveStream {
  status: Status | null;
  history: Sample[]; // the last two minutes, oldest first
  events: EventItem[]; // newest first: cross-camera links and camera state changes
  closedTracks: number; // tracks finished since the page was opened
  connected: boolean;
}

const HISTORY = 120;

function sample(s: Status): Sample {
  const fps = Object.values(s.cameras ?? {}).reduce((a, c) => a + (c.processed_fps ?? 0), 0);
  return { t: s.time ?? Date.now() / 1000, fps, gpu: s.gpu?.util_percent ?? null, cpu: s.process_cpu_percent ?? null };
}

const KEEP = 200;

/** One Server-Sent-Events connection: status once per second, events as they are committed. The browser
 *  reconnects by itself (the server sends `retry: 3000`); the initial events come from the REST endpoint. */
export function useLiveStream(): LiveStream {
  const [status, setStatus] = useState<Status | null>(null);
  const [history, setHistory] = useState<Sample[]>([]);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [closedTracks, setClosed] = useState(0);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let alive = true;
    getJSON<Status>("/api/status").then((s) => alive && setStatus(s)).catch(() => {});
    getJSON<EventItem[]>("/api/events", { limit: 50 })
      .then((ev) => alive && setEvents((cur) => (cur.length ? cur : ev)))
      .catch(() => {});
    const es = new EventSource(apiUrl("/api/stream"));
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.addEventListener("status", (m) => {
      const st = JSON.parse((m as MessageEvent).data) as Status;
      setStatus(st);
      setHistory((h) => [...h, sample(st)].slice(-HISTORY));
    });
    es.addEventListener("event", (m) => {
      const e = JSON.parse((m as MessageEvent).data) as EventItem & { kind?: string };
      if (e.type === "track_close") setClosed((n) => n + 1);
      if (e.type === "identity_link" || e.type === "camera_state") setEvents((cur) => [e, ...cur].slice(0, KEEP));
    });
    return () => {
      alive = false;
      es.close();
    };
  }, []);
  return { status, history, events, closedTracks, connected };
}

/** Fetched once: what the server is (health: examples, prompt; system: models, cameras, hardware). */
export function useOnce<T>(path: string): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    getJSON<T>(path).then((d) => alive && setData(d)).catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [path]);
  return { data, error };
}

export const useHealth = () => useOnce<Health>("/api/health");
export const useSystem = () => useOnce<SystemInfo>("/api/system");

/** Seconds since the epoch, refreshed every `ms` (relative times such as "12 s ago"). */
export function useNow(ms = 5000): number {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

/** Re-runs `fn` every `ms` while the page is visible. */
export function useInterval(fn: () => void, ms: number, deps: unknown[] = []): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    ref.current();
    const id = window.setInterval(() => document.visibilityState === "visible" && ref.current(), ms);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, ...deps]);
}
