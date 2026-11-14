// Typed client for the FastAPI service (src/lgep/api/app.py). All calls go to the same origin under /api.

export interface CameraStatus {
  camera_id: string;
  url: string;
  state: string; // connecting | streaming | reconnecting | ended | stopped
  stalled: boolean;
  input_fps: number;
  frames_skipped: number;
  connects: number;
  last_error: string | null;
  last_frame_age_s: number | null;
  frame_size: [number, number] | null;
  processed_fps?: number | null;
  latency_ms_p50?: number | null;
  latency_ms_p90?: number | null;
  detector_ms_p50?: number | null;
  active_tracks?: number;
}

export interface Status {
  live: boolean;
  run_id: number | null;
  uptime_s: number;
  index_tracks: number;
  time?: number;
  cameras?: Record<string, CameraStatus>;
  pipeline?: { target_process_fps: number; phrases: string[]; callback_ms_mean: number };
  reid?: {
    global_identities: number; multi_camera_identities: number; embed_ms_per_frame: number; crops_embedded?: number;
    submitted?: number; dropped?: number; queued?: number; lag_ms_p50?: number | null; lag_ms_p90?: number | null;
  } | null;
  store: { size_mb: number; queued: number; dropped: number; written: number; last_commit_ms: number; failed?: number };
  live_hub?: { encodes: number; frames_served: number; encode_ms_mean: number };
  gpu?: { util_percent: number; memory_used_mib: number; power_w?: number | null } | null;
  process_cpu_percent?: number;
  last_hour?: { tracks: number; open_tracks: number; identities: number; multi_camera_identities: number; identity_links: number };
  sse_clients?: number;
}

export interface Track {
  id: number;
  run_id: number;
  key: string;
  camera_id: string;
  local_id: number;
  category: string;
  t_start: number;
  t_end: number;
  duration_s: number;
  frames: number;
  max_score: number | null;
  closed: boolean;
  identity_id: number | null;
  global_id: number | null;
  identity_cameras: string[];
  decision: string | null;
  link_score: number | null;
  has_thumbnail: boolean;
}

export interface EventItem {
  id?: number;
  type: string; // identity_link | camera_state | track_open | track_close
  t: number;
  camera_id: string | null;
  track_id?: number | null;
  identity_id?: number | null;
  data: Record<string, unknown>;
}

export interface Identity {
  id: number;
  run_id: number;
  global_id: number;
  t_start: number;
  t_end: number;
  cameras: string[];
  n_tracks: number;
  tracks?: Track[];
  links?: EventItem[];
}

export interface Parsed {
  categories: string[];
  object_colours: string[];
  clothing: { item: string | null; region: string; colour: string }[];
  carried: string[];
  cameras: string[];
  time: { last_s: number | null; after: string | null; before: string | null } | null;
  free_text: string;
  vehicle_types?: string[];
}

export interface SearchHit {
  rank: number;
  score: number;
  text: number;
  attributes: number;
  track: string;
  camera: string;
  global_id: number | null;
  category: string;
  seen_in_cameras: string[];
  start: string;
  duration_s: number;
  track_id: number | null;
  run_id: number | null;
  identity_id: number | null;
  has_thumbnail: boolean;
  t_start: number;
  t_end: number;
}

export interface SearchResponse {
  query: string;
  parsed: Parsed;
  indexed_tracks: number;
  weights: { text: number; attributes: number };
  hits: SearchHit[];
  timing_ms: { parse: number; search: number; total: number };
}

export interface Health {
  status: string;
  live: boolean;
  run_id: number | null;
  version: string;
  search: boolean;
  auth: boolean;
  query_examples: string[];
  prompt: string | string[] | null;
}

export interface ModelInfo {
  model_id?: string;
  backend?: string;
  dtype?: string;
  imgsz?: number;
  device?: string;
  parameters_millions?: number;
  [k: string]: unknown;
}

export interface SystemInfo {
  live: boolean;
  run_id: number | null;
  environment: {
    os?: string; python?: string; cpu?: string; cpu_logical_cores?: number; ram_gib?: number; torch?: string;
    torch_cuda_build?: string; gpu?: { name?: string; driver?: string; memory_total?: string; power_limit?: string } | null;
  };
  store: { retention_hours: number | null; thumbnails: boolean; points_every_s: number; batch_s: number };
  search: { weights: { text: number; attributes: number }; embedder: ModelInfo | null; vehicle_type_check: boolean } | null;
  detector?: ModelInfo;
  pipeline?: { phrases: string[]; process_fps: number; batch_size: number; tracker: string; track_thresh: number;
    phrase_groups: Record<string, string[]> };
  reid?: { asynchronous: boolean; close_after_s: number; appearance_threshold: number; embedder: ModelInfo } | null;
  cameras?: { camera_id: string; url: string; kind: string; source?: string | null }[];
  opencv_threads?: number | null;
}

export interface Timeline {
  since: number;
  until: number;
  bucket_s: number;
  buckets: number[];
  tracks_started: Record<string, number[]>;
  identity_links: number[];
}

// ------------------------------------------------------------------------------------------- token
// An API token (if the server requires one) is passed once as ?token=... and kept for this browser tab.
function initToken(): string | null {
  try {
    const q = new URLSearchParams(window.location.search).get("token");
    if (q) sessionStorage.setItem("lgep-token", q);
    return q ?? sessionStorage.getItem("lgep-token");
  } catch {
    return null;
  }
}
const token = initToken();

export function apiUrl(path: string, params: Record<string, string | number | boolean | undefined | null> = {}): string {
  const u = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  if (token) u.searchParams.set("token", token); // <img> and EventSource cannot send headers
  return u.pathname + u.search;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function check(r: Response): Promise<Response> {
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const body = await r.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg);
  }
  return r;
}

export async function getJSON<T>(path: string, params: Record<string, string | number | boolean | undefined | null> = {},
                                 signal?: AbortSignal): Promise<T> {
  return (await check(await fetch(apiUrl(path, params), { signal }))).json();
}

export async function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(apiUrl(path), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  return (await check(r)).json();
}

export const thumbUrl = (trackId: number) => apiUrl(`/api/tracks/${trackId}/thumbnail.jpg`);
