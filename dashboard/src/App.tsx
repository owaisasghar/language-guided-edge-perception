import { useCallback, useEffect, useState } from "react";
import { IdentityDrawer } from "./components/IdentityDrawer";
import { LiveView } from "./components/Live";
import { SearchView, StatTiles } from "./components/Search";
import { TimelineView } from "./components/Timeline";
import { useLiveStream } from "./hooks";

type Tab = "live" | "search" | "timeline";

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try { return localStorage.getItem("lgep-theme") ?? "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    const el = document.documentElement;
    if (theme === "auto") el.removeAttribute("data-theme"); else el.setAttribute("data-theme", theme);
    try { localStorage.setItem("lgep-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const next = () => setTheme((t) => (t === "auto" ? "light" : t === "light" ? "dark" : "auto"));
  return [theme, next];
}

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash.slice(1) as Tab) || "live");
  const [identity, setIdentity] = useState<number | null>(null);
  const [theme, nextTheme] = useTheme();
  const { status, events, closedTracks, connected } = useLiveStream();
  const closeIdentity = useCallback(() => setIdentity(null), []);

  useEffect(() => { history.replaceState(null, "", `#${tab}`); }, [tab]);
  const cameras = Object.keys(status?.cameras ?? {}).sort();
  const live = status?.live ?? true;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Edge Perception <small>{live ? `live · run ${status?.run_id ?? "–"}` : "replay of stored runs"}</small></div>
        <nav className="tabs" role="tablist">
          {(["live", "search", "timeline"] as Tab[]).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>
              {t === "live" ? "Live" : t === "search" ? "Search" : "Timeline"}
            </button>
          ))}
        </nav>
        <button className="icon-btn" onClick={nextTheme} title="Colour theme">Theme: {theme}</button>
      </header>
      <StatTiles status={status} closedTracks={closedTracks} connected={connected} />
      {tab === "live" && <LiveView cameras={status?.cameras} events={events} onIdentity={setIdentity} />}
      {tab === "search" && <SearchView onIdentity={setIdentity} />}
      {tab === "timeline" && <TimelineView cameras={cameras} live={live} onIdentity={setIdentity} />}
      {identity != null && <IdentityDrawer id={identity} onClose={closeIdentity} />}
    </div>
  );
}
