import { describe, expect, it } from "vitest";
import { bucketSeconds, cameraHealth, compact, fmtDuration, fmtGap, niceStep, packRows, seqStep, sparkPath, timeTicks } from "./charts";

describe("packRows", () => {
  it("puts overlapping intervals on separate rows and reuses free rows", () => {
    const items = [
      { t_start: 0, t_end: 10 },
      { t_start: 5, t_end: 8 }, // overlaps the first
      { t_start: 9, t_end: 12 }, // row 1 is free again (ended at 8)
      { t_start: 11, t_end: 13 }, // row 0 is free (ended at 10)
    ];
    expect(packRows(items)).toEqual({ rows: [0, 1, 1, 0], count: 2 });
  });
  it("respects the gap and keeps input order in the result", () => {
    const items = [
      { t_start: 10.5, t_end: 11 },
      { t_start: 0, t_end: 10 },
    ];
    expect(packRows(items, 1).rows).toEqual([1, 0]); // 10 + 1 > 10.5: needs a new row
    expect(packRows([]).count).toBe(0);
  });
});

describe("time scales", () => {
  it("chooses nice steps and ticks inside the window", () => {
    expect(niceStep(300, 6)).toBe(60);
    expect(niceStep(7, 6)).toBe(2);
    const ticks = timeTicks(1000.4, 1012, 6);
    expect(ticks).toEqual([1002, 1004, 1006, 1008, 1010, 1012]);
    expect(bucketSeconds(3600, 60)).toBe(60);
    expect(bucketSeconds(300, 60)).toBe(5);
  });
});

describe("seqStep", () => {
  it("maps counts to 1..7 with zero left empty", () => {
    expect(seqStep(0, 10)).toBe(0);
    expect(seqStep(10, 10)).toBe(7);
    expect(seqStep(1, 100)).toBe(1);
    expect(seqStep(25, 100)).toBeGreaterThan(seqStep(4, 100));
  });
});

describe("cameraHealth", () => {
  it("always pairs the level with an icon and a label", () => {
    expect(cameraHealth({ state: "streaming" })).toEqual({ level: "good", icon: "✓", label: "Live" });
    expect(cameraHealth({ state: "streaming", stalled: true }).label).toBe("Stalled");
    expect(cameraHealth({ state: "reconnecting" }).level).toBe("serious");
    expect(cameraHealth({ state: "stopped" })).toEqual({ level: "critical", icon: "×", label: "Stopped" });
    expect(cameraHealth(undefined).level).toBe("critical");
  });
});

describe("formatting", () => {
  it("formats durations and compact numbers", () => {
    expect(fmtDuration(3.21)).toBe("3.2 s");
    expect(fmtDuration(75)).toBe("1 min 15 s");
    expect(compact(1284)).toBe("1,284");
    expect(compact(12_900)).toBe("12.9K");
    expect(compact(4_200_000)).toBe("4.2M");
  });
});


describe("sparkPath and gaps", () => {
  it("draws through the points, skipping missing ones, from a zero baseline", () => {
    expect(sparkPath([0, 10], 100, 20, 0)).toBe("M0.0,20.0 L100.0,0.0");
    expect(sparkPath([5, null, 10], 100, 20, 0)).toBe("M0.0,10.0 L100.0,0.0");
    expect(sparkPath([1], 100, 20)).toBe("");
  });
  it("formats the time between two appearances", () => {
    expect(fmtGap(10, 22)).toBe("+12 s");
    expect(fmtGap(10, 5)).toBe("at the same time");
  });
});
