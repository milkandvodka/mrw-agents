// Pure logic shared by the UI (no React, no DOM) so it can be tested with `node --test`.

/** Split an SSE text buffer into complete `data:` events and the unfinished remainder. */
export function parseSse(buffer: string): { events: Record<string, unknown>[]; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const events = parts
    .map((p) => p.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join(""))
    .filter(Boolean)
    .map((d) => JSON.parse(d) as Record<string, unknown>);
  return { events, rest };
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-11-15" -> "15 Nov 2026", in the data's own calendar (no time zone shift). */
export function longDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return y && m && d ? `${d} ${MONTHS[m - 1]} ${y}` : iso;
}

/** "2026-10-23" -> "23 Oct". */
export function shortDate(iso: string): string {
  const [, m, d] = iso.split("-").map(Number);
  return m && d ? `${d} ${MONTHS[m - 1]}` : iso;
}

/** Whole days from one ISO date to another (UTC, so no daylight-saving drift). */
export function daysBetween(from: string, to: string): number {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86400000);
}

/** Minutes as "45 min" or "2h 5m", the way the agent writes them. */
export function hm(minutes: number): string {
  const h = Math.floor(Math.round(minutes) / 60), m = Math.round(minutes) % 60;
  return h ? `${h}h ${m}m` : `${m} min`;
}
