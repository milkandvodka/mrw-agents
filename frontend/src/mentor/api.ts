import { parseSse } from "./lib";

const API = "/mentor/api"; // this agent's half of the shared server

export interface Overview {
  students: number;
  batches: string[];
  as_of: string;
  examples: string[];
  has_key: boolean;
}

export type Integrity = "confirmed" | "open" | "cleared" | null;

export interface StudentRow {
  id: string;
  name: string;
  batch: string;
  joined: string;
  last_active: string | null;
  active_30: number;
  lectures_started: number;
  lectures_full: number;
  tests_taken: number;
  tests_held: number;
  latest_gt: { test: string; rank: number; of: number } | null;
  accuracy: number | null;
  integrity: Integrity;
}

export interface StudentDetail {
  id: string;
  name: string;
  batch: string;
  joined: string;
  mentor: string;
  last_active: string | null;
  weeks: { start: string; end: string; days: number; minutes: number }[];
  tests: { test: string; date: string; kind: string; score: number; max: number; right: number; wrong: number; skipped: number;
    rank: number; of: number; minutes: number; limit: number; tabs: number }[];
  missed: string[];
  subjects: { subject: string; accuracy: number | null; answered: number; batch_accuracy: number | null;
    lectures: { title: string; pct: number | null; first: string | null }[] }[];
  reviews: { test: string; signal: string; outcome: string; opened: string; decided: string | null; note: string }[];
}

export async function getStudents(): Promise<StudentRow[]> {
  const r = await fetch(`${API}/students`);
  if (!r.ok) throw new Error(`Server returned ${r.status}`);
  return r.json();
}

export async function getStudent(id: string): Promise<StudentDetail> {
  const r = await fetch(`${API}/students/${id}`);
  if (!r.ok) throw new Error(`Server returned ${r.status}`);
  return r.json();
}

export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool_end"; id: string; name: string; summary: string; content: string }
  | { type: "error"; message: string }
  | { type: "done" };

export async function getOverview(): Promise<Overview> {
  const r = await fetch(`${API}/overview`);
  if (!r.ok) throw new Error(`Server returned ${r.status}`);
  return r.json();
}

export async function streamChat(sessionId: string, message: string, onEvent: (e: AgentEvent) => void, signal: AbortSignal) {
  const r = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
  if (!r.ok || !r.body) {
    onEvent({ type: "error", message: `The server couldn't take that request (${r.status}). Try a shorter message.` });
    return;
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let rest = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseSse(rest + value);
    rest = parsed.rest;
    parsed.events.forEach((e) => onEvent(e as AgentEvent));
  }
}

export function resetSession(sessionId: string) {
  return fetch(`${API}/session/${sessionId}`, { method: "DELETE" }).catch(() => undefined);
}
