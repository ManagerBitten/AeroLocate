import { describe, expect, it } from "vitest";
import { formatArea, formatDistance, formatProbability, zonedToUtcSeconds } from "./format";

describe("zonedToUtcSeconds", () => {
  it("handles a zone with daylight saving", () => {
    expect(zonedToUtcSeconds("2025-06-15", "16:06:40", "Europe/London")).toBe(1750000000);
    expect(zonedToUtcSeconds("2025-01-15", "12:00:00", "Europe/London")).toBe(Date.UTC(2025, 0, 15, 12) / 1000);
  });

  it("handles a zone west of UTC", () => {
    expect(zonedToUtcSeconds("2025-06-15", "08:00:00", "America/Los_Angeles")).toBe(Date.UTC(2025, 5, 15, 15) / 1000);
  });

  it("handles a half-hour offset", () => {
    expect(zonedToUtcSeconds("2025-06-15", "12:00:00", "Asia/Kolkata")).toBe(Date.UTC(2025, 5, 15, 6, 30) / 1000);
  });
});

describe("formatters", () => {
  it("does not overstate certainty", () => {
    expect(formatProbability(0.999)).toBe(">99%");
    expect(formatProbability(0.001)).toBe("<1%");
    expect(formatProbability(0.42)).toBe("42%");
  });

  it("uses readable units", () => {
    expect(formatDistance(1500)).toBe("1.5 km");
    expect(formatDistance(430.4)).toBe("430 m");
    expect(formatArea(0.03)).toBe("3.0 ha");
    expect(formatArea(12.34)).toBe("12.3 km²");
  });
});
