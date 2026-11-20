import { parseSse } from "./lib";
import type { Finding, Planted } from "./lib";

const API = "/qbank/api"; // this agent's half of the shared server

// a chip on the first screen: label is what it says, prompt is what it sends (a pasted MCQ, or a question about the bank)
export interface Suggestion {
  label: string;
  prompt: string;
}

export interface Overview {
  questions: number;
  topics: number;
  counts: Record<string, number>;
  suggestions: Suggestion[];
  examples: string[];
  planted: Planted[];
  has_key: boolean;
}

export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; id: string; name: string; args: Record<string, unknown> }
  | { type: "progress"; id: string; tool: string; done: number; total: number }
  | { type: "tool_end"; id: string; name: string; summary: string; content: string; findings: Finding[] }
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
