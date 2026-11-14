// A small set of stroke icons (24 x 24, currentColor), drawn for this dashboard: no icon library dependency.
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 18, children, ...rest }: P & { children: React.ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...rest}>
      {children}
    </svg>
  );
}

export const IconCamera = (p: P) => (
  <Svg {...p}><rect x="2.5" y="6.5" width="13" height="11" rx="2.5" /><path d="M15.5 10.5 21 7.5v9l-5.5-3" /></Svg>
);
export const IconSearch = (p: P) => (
  <Svg {...p}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></Svg>
);
export const IconTimeline = (p: P) => (
  <Svg {...p}><path d="M3.5 6.5h9M3.5 12h14M3.5 17.5h6" /><circle cx="15.5" cy="6.5" r="1.6" /><circle cx="20" cy="12" r="1.6" />
    <circle cx="12.5" cy="17.5" r="1.6" /></Svg>
);
export const IconChip = (p: P) => (
  <Svg {...p}><rect x="6" y="6" width="12" height="12" rx="2.5" /><rect x="9.5" y="9.5" width="5" height="5" rx="1" />
    <path d="M9 2.5V6M15 2.5V6M9 18v3.5M15 18v3.5M2.5 9H6M2.5 15H6M18 9h3.5M18 15h3.5" /></Svg>
);
export const IconSun = (p: P) => (
  <Svg {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" /></Svg>
);
export const IconMoon = (p: P) => (
  <Svg {...p}><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" /></Svg>
);
export const IconMonitor = (p: P) => (
  <Svg {...p}><rect x="3" y="4" width="18" height="12.5" rx="2" /><path d="M8.5 20h7M12 16.5V20" /></Svg>
);
export const IconLink = (p: P) => (
  <Svg {...p}><path d="M10 14a4.5 4.5 0 0 0 6.4 0l3-3A4.5 4.5 0 0 0 13 4.6l-1.3 1.3" />
    <path d="M14 10a4.5 4.5 0 0 0-6.4 0l-3 3A4.5 4.5 0 0 0 11 19.4l1.3-1.3" /></Svg>
);
export const IconAlert = (p: P) => (
  <Svg {...p}><path d="M10.3 4.2 2.8 17.5A2 2 0 0 0 4.5 20.5h15a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" />
    <path d="M12 9.5v4.5M12 17.2v.1" /></Svg>
);
export const IconCheck = (p: P) => (<Svg {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></Svg>);
export const IconClock = (p: P) => (<Svg {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></Svg>);
export const IconX = (p: P) => (<Svg {...p}><path d="M6 6l12 12M18 6 6 18" /></Svg>);
export const IconDatabase = (p: P) => (
  <Svg {...p}><ellipse cx="12" cy="6" rx="7.5" ry="3" /><path d="M4.5 6v12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6" />
    <path d="M4.5 12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3" /></Svg>
);
export const IconGauge = (p: P) => (
  <Svg {...p}><path d="M4 17a8.5 8.5 0 1 1 16 0" /><path d="m12 13.5 4-4.5" /><circle cx="12" cy="14" r="1.3" /></Svg>
);
export const IconLayers = (p: P) => (
  <Svg {...p}><path d="m12 3.5 9 4.8-9 4.8-9-4.8 9-4.8Z" /><path d="m3 12.8 9 4.8 9-4.8" /></Svg>
);
export const IconGrid = (p: P) => (
  <Svg {...p}><rect x="3.5" y="3.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
    <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.5" /></Svg>
);
export const IconExpand = (p: P) => (<Svg {...p}><path d="M14.5 3.5h6v6M9.5 20.5h-6v-6M20.5 3.5 14 10M3.5 20.5 10 14" /></Svg>);
export const IconUser = (p: P) => (<Svg {...p}><circle cx="12" cy="8" r="4" /><path d="M4.5 20.5a7.5 7.5 0 0 1 15 0" /></Svg>);
export const IconCar = (p: P) => (
  <Svg {...p}><path d="M5 16.5H3.5v-4l2-5h13l2 5v4H19" /><path d="M3.5 12.5h17" /><circle cx="7.5" cy="16.5" r="2" />
    <circle cx="16.5" cy="16.5" r="2" /><path d="M9.5 16.5h5" /></Svg>
);
export const IconArrowRight = (p: P) => (<Svg {...p}><path d="M5 12h14M13.5 6.5 19 12l-5.5 5.5" /></Svg>);
export const IconChevron = (p: P) => (<Svg {...p}><path d="m9 5.5 6.5 6.5L9 18.5" /></Svg>);
export const IconOffline = (p: P) => (
  <Svg {...p}><path d="M3 3l18 18M8.5 16.5a5 5 0 0 1 6.5-.5M5 12.5a10 10 0 0 1 4-2.4M14.8 10.3A10 10 0 0 1 19 12.5M2 9a15 15 0 0 1 3.4-2.2M11 5a15 15 0 0 1 11 4" />
    <path d="M12 20h.01" /></Svg>
);
export const IconSpark = (p: P) => (<Svg {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" /></Svg>);

/** The product mark: a lens (camera) whose iris is a small network of linked nodes (cameras, identities). */
export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <rect x="1" y="1" width="30" height="30" rx="8" fill="var(--accent)" />
      <circle cx="16" cy="16" r="8.5" fill="none" stroke="#fff" strokeWidth="2" opacity="0.9" />
      <path d="M12.3 18.2 16 12.4l3.7 5.8" fill="none" stroke="#fff" strokeWidth="1.6" strokeLinejoin="round" />
      <circle cx="16" cy="12.4" r="1.9" fill="#fff" /><circle cx="12.3" cy="18.2" r="1.9" fill="#fff" />
      <circle cx="19.7" cy="18.2" r="1.9" fill="#fff" />
    </svg>
  );
}
