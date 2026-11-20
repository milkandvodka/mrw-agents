// Pure logic shared by the UI (no React, no DOM) so it can be tested with `node --test`.

export type Severity = "high" | "medium" | "low";
export type Status = "open" | "confirmed" | "dismissed";

export interface Question {
  id: string;
  question: string;
  options: string[];
  answer: number;
  explanation: string;
  topic: string;
}

export interface Finding {
  key: string;
  check: string;
  severity: Severity;
  source: "exact" | "ai" | "agent";
  question_id: string;
  related_ids: string[];
  title: string;
  detail: string;
  reason?: string; // AI findings: the reviewer's reason on its own, for the card
  suggested?: number | null;
  question: Question;
}

export interface Item extends Finding {
  status: Status;
  addedAt: number;
}

export type Queue = Record<string, Item>;

export interface Planted {
  type: string;
  ids: string[];
  topic?: string;
}

export const LETTERS = ["A", "B", "C", "D"];

export const PLANT_NAMES: Record<string, string> = {
  key_conflict: "Duplicate copy with a different key",
  duplicate_options: "An option repeated",
  blank_option: "An option blanked",
  explanation_conflict: "Key changed, explanation left as is (AI review)",
  ai_only_wrong_key: "Silent wrong key (AI review)",
};

export function mergeFindings(queue: Queue, findings: Finding[], now: number): { queue: Queue; added: number } {
  const next: Queue = { ...queue };
  let added = 0;
  for (const f of findings) {
    if (!next[f.key]) {
      next[f.key] = { ...f, status: "open", addedAt: now };
      added++;
    }
  }
  return { queue: next, added };
}

export function scorecard(queue: Queue, planted: Planted[]) {
  const flagged = new Set<string>();
  for (const f of Object.values(queue)) [f.question_id, ...f.related_ids].forEach((id) => flagged.add(id));
  const plantedIds = new Set(planted.flatMap((p) => p.ids));
  const byType: Record<string, [number, number]> = {};
  let caught = 0;
  for (const p of planted) {
    const hit = p.ids.some((id) => flagged.has(id));
    caught += hit ? 1 : 0;
    const t = (byType[p.type] ??= [0, 0]);
    t[0] += hit ? 1 : 0;
    t[1] += 1;
  }
  const real = Object.values(queue).filter((f) => ![f.question_id, ...f.related_ids].some((id) => plantedIds.has(id))).length;
  return { caught, total: planted.length, byType, real };
}

function cell(v: string): string {
  const safe = v && "=+-@\t\r".includes(v[0]) ? `'${v}` : v; // stop spreadsheets from running cell text as a formula
  return `"${safe.replaceAll('"', '""')}"`;
}

export function toCsv(items: Item[]): string {
  const head = ["status", "severity", "evidence", "check", "question_id", "related_ids", "title", "detail", "topic", "question"];
  const rows = items.map((f) =>
    [f.status, f.severity, f.source === "exact" ? "exact match" : "AI opinion", f.check, f.question_id, f.related_ids.join(" "),
      f.title, f.detail, f.question.topic, f.question.question].map(cell).join(","));
  return [head.join(","), ...rows].join("\n");
}

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
