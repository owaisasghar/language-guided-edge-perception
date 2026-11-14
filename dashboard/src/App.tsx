import { useCallback, useEffect, useState, type ReactNode } from "react";
import { IdentityDrawer } from "./components/IdentityDrawer";
import { IconAlert, IconCamera, IconChip, IconMonitor, IconMoon, IconSearch, IconSun, IconTimeline, Logo } from "./components/Icons";
import { LiveView } from "./components/Live";
import { SearchView } from "./components/Search";
import { SystemView } from "./components/System";
import { TimelineView } from "./components/Timeline";
import { useHealth, useLiveStream, useSystem } from "./hooks";
import { fmtDuration } from "./lib/charts";

type Page = "live" | "search" | "timeline" | "system";
type Theme = "auto" | "light" | "dark";

const PAGES: { id: Page; label: string; icon: ReactNode }[] = [
  { id: "live", label: "Live", icon: <IconCamera /> },
  { id: "search", label: "Search", icon: <IconSearch /> },
  { id: "timeline", label: "Timeline", icon: <IconTimeline /> },
  { id: "system", label: "System", icon: <IconChip /> },
];

function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try { return (localStorage.getItem("lgep-theme") as Theme) || "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "auto") el.removeAttribute("data-theme"); else el.setAttribute("data-theme", theme);
    try { localStorage.setItem("lgep-theme", theme); } catch { /* storage unavailable: the choice lasts this visit */ }
  }, [theme]);
  return [theme, setTheme];
}

function pageFromHash(): Page {
  const h = window.location.hash.slice(1);
  return (PAGES.some((p) => p.id === h) ? h : "live") as Page;
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [identity, setIdentity] = useState<number | null>(() => { // deep link: ?identity=12
    const v = Number(new URLSearchParams(window.location.search).get("identity"));
    return Number.isInteger(v) && v > 0 ? v : null;
  });
  const [theme, setTheme] = useTheme();
  const [graceOver, setGraceOver] = useState(false);
  const { status, history, events, closedTracks, connected } = useLiveStream();
  const health = useHealth().data;
  const system = useSystem().data;
  const closeIdentity = useCallback(() => setIdentity(null), []);

  useEffect(() => { window.history.replaceState(null, "", `${window.location.search}#${page}`); window.scrollTo(0, 0); }, [page]);
  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    const t = window.setTimeout(() => setGraceOver(true), 5000);
    return () => { window.removeEventListener("hashchange", onHash); window.clearTimeout(t); };
  }, []);
  const titles: Record<Page, string> = { live: "Live", search: "Search", timeline: "Timeline", system: "System" };
  useEffect(() => { document.title = `${titles[page]} · Edge Perception`; });

  const cameras = Object.keys(status?.cameras ?? {}).sort();
  const live = status?.live ?? health?.live ?? true;
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <div>
            <div className="brand-name">Edge Perception</div>
            <div className="brand-sub">language-guided multi-camera</div>
          </div>
        </div>
        <nav aria-label="Pages">
          <div className="nav-label">Workspace</div>
          <div className="nav">
            {PAGES.map((p) => (
              <button key={p.id} className="nav-item" aria-current={page === p.id ? "page" : undefined} onClick={() => setPage(p.id)}
                      title={p.label}>
                {p.icon}<span>{p.label}</span>
              </button>
            ))}
          </div>
        </nav>
        <div className="sidebar-foot">
          <div className="conn" aria-live="polite">
            <div className="conn-row"><span>Live updates</span>
              <span className={`status ${connected ? "good" : "serious"}`}><span className="dot" aria-hidden="true">{connected ? "✓" : "!"}</span>{connected ? "connected" : "reconnecting"}</span></div>
            <div className="conn-row"><span>Mode</span><b>{live ? `live · run ${status?.run_id ?? "–"}` : "replay"}</b></div>
            <div className="conn-row"><span>Uptime</span><b className="num">{status ? fmtDuration(status.uptime_s) : "–"}</b></div>
            <div className="conn-row"><span>Version</span><b className="num">{health?.version ?? "–"}</b></div>
          </div>
          <div className="theme-switch" role="group" aria-label="Colour theme">
            <button aria-pressed={theme === "auto"} onClick={() => setTheme("auto")} title="Follow the system"><IconMonitor size={16} /><span className="sr-only">System</span></button>
            <button aria-pressed={theme === "light"} onClick={() => setTheme("light")} title="Light"><IconSun size={16} /><span className="sr-only">Light</span></button>
            <button aria-pressed={theme === "dark"} onClick={() => setTheme("dark")} title="Dark"><IconMoon size={16} /><span className="sr-only">Dark</span></button>
          </div>
        </div>
      </aside>
      <main className="main">
        {graceOver && !connected && (
          <div className="banner warn" role="status"><IconAlert size={16} />Live updates are paused: the connection to the server was lost. Reconnecting…</div>
        )}
        {page === "live" && <LiveView status={status} history={history} events={events} closedTracks={closedTracks} system={system} onIdentity={setIdentity} />}
        {page === "search" && <SearchView health={health} onIdentity={setIdentity} />}
        {page === "timeline" && <TimelineView cameras={cameras} live={live} onIdentity={setIdentity} />}
        {page === "system" && <SystemView status={status} system={system} />}
      </main>
      {identity != null && <IdentityDrawer id={identity} onClose={closeIdentity} />}
    </div>
  );
}
