import { useEffect, useRef, useState } from "react";
import { apiUrl, getJSON, type EventItem, type Status } from "./api";

export interface LiveStream {
  status: Status | null;
  events: EventItem[]; // newest first: cross-camera links and camera state changes
  closedTracks: number; // tracks finished since the page was opened
  connected: boolean;
}

const KEEP = 200;

/** One Server-Sent-Events connection: status once per second, events as they are committed. The browser
 *  reconnects by itself (the server sends `retry: 3000`); the initial events come from the REST endpoint. */
export function useLiveStream(): LiveStream {
  const [status, setStatus] = useState<Status | null>(null);
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
    es.addEventListener("status", (m) => setStatus(JSON.parse((m as MessageEvent).data)));
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
  return { status, events, closedTracks, connected };
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
