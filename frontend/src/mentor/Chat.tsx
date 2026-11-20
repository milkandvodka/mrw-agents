import {
  Activity, ArrowUp, ChartColumn, Check, ChevronRight, CircleAlert, CirclePlay, ClipboardList, Database, GraduationCap, Layers,
  ListOrdered, LoaderCircle, Search, ShieldAlert, Square, UserRound, X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { longDate } from "./lib";

export type ToolPart = {
  kind: "tool";
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  content: string;
};
export type Part = { kind: "text"; text: string } | ToolPart;
export type Turn = { role: "user"; text: string } | { role: "assistant"; parts: Part[]; error?: string };

const TOOLS: Record<string, [string, string, LucideIcon]> = { // running label, done label, icon
  overview: ["Reading the batches and tests", "Read the batches and tests", Layers],
  find_students: ["Finding students", "Found students", Search],
  student_report: ["Reading the student's report", "Read the student's report", UserRound],
  activity: ["Reading app use", "Read app use", Activity],
  lectures: ["Reading lecture history", "Read lecture history", CirclePlay],
  tests: ["Reading test results", "Read test results", ClipboardList],
  performance: ["Reading subject performance", "Read subject performance", ChartColumn],
  integrity: ["Checking test integrity", "Checked test integrity", ShieldAlert],
  rank_students: ["Ranking students", "Ranked students", ListOrdered],
  run_sql: ["Querying the data", "Queried the data", Database],
};

function describe(args: Record<string, unknown>): string {
  const bits = ["student", "name", "test", "subject", "batch"].filter((k) => args[k]).map((k) => String(args[k]));
  if (args.metric) bits.push(String(args.metric).replaceAll("_", " "));
  return bits.length ? ` · ${bits.join(", ")}` : "";
}

function ToolStep({ p }: { p: ToolPart }) {
  const [open, setOpen] = useState(false);
  const [running, done, Icon] = TOOLS[p.name] ?? [p.name, p.name, Layers];
  return (
    <div>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} disabled={!p.content}
        className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left enabled:hover:bg-canvas/70 outline-none focus-visible:bg-canvas">
        <span className="shrink-0 size-6 grid place-items-center">
          {p.status === "running" ? (
            <LoaderCircle size={16} className="animate-spin text-brand-600" />
          ) : p.status === "done" ? (
            <span className="size-5 rounded-full bg-ok-50 text-ok-700 grid place-items-center"><Check size={12} strokeWidth={3} /></span>
          ) : (
            <span className="size-5 rounded-full bg-bad-50 text-bad-700 grid place-items-center"><X size={12} strokeWidth={3} /></span>
          )}
        </span>
        <Icon size={15} className="shrink-0 text-faint" />
        <span className="min-w-0 flex-1 text-[13.5px] text-ink truncate">
          {(p.status === "running" ? running : done) + describe(p.args)}
          {p.status === "running" ? "…" : ""}
        </span>
        {p.content && <ChevronRight size={15} className={`shrink-0 text-faint transition ${open ? "rotate-90" : ""}`} />}
      </button>
      {open && p.content && (
        <pre className="mx-3.5 mb-3 max-h-72 overflow-auto rounded-lg bg-canvas border border-line p-3 text-[12px] leading-relaxed text-body whitespace-pre-wrap font-sans">
          {p.content}
        </pre>
      )}
    </div>
  );
}

function AssistantTurn({ turn, live }: { turn: Extract<Turn, { role: "assistant" }>; live: boolean }) {
  const blocks: (Part | ToolPart[])[] = []; // consecutive tool calls form one step list
  for (const p of turn.parts) {
    const last = blocks[blocks.length - 1];
    if (p.kind === "tool" && Array.isArray(last)) last.push(p);
    else blocks.push(p.kind === "tool" ? [p] : p);
  }
  const lastPart = turn.parts[turn.parts.length - 1];
  const waiting = live && !turn.error && (!lastPart || (lastPart.kind === "tool" && lastPart.status !== "running"));
  return (
    <div className="flex gap-3">
      <div className="size-7 shrink-0 rounded-lg bg-brand-50 text-brand-700 grid place-items-center mt-0.5">
        <GraduationCap size={15} />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {blocks.map((b, i) =>
          Array.isArray(b) ? (
            <div key={i} className="rounded-xl border border-line bg-white divide-y divide-line overflow-hidden">
              {b.map((p) => <ToolStep key={p.id} p={p} />)}
            </div>
          ) : b.kind === "text" ? (
            <div key={i} className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{b.text}</ReactMarkdown></div>
          ) : null,
        )}
        {waiting && (
          <div className="flex items-center gap-2 text-[13px] text-muted">
            <span className="dots inline-flex gap-0.5" aria-hidden>
              <span className="size-1.5 rounded-full bg-brand-500" /><span className="size-1.5 rounded-full bg-brand-500" /><span className="size-1.5 rounded-full bg-brand-500" />
            </span>
            Thinking
          </div>
        )}
        {turn.error && (
          <div role="alert" className="flex items-start gap-2 rounded-lg bg-bad-50 border border-bad-200 text-bad-700 text-[13px] px-3 py-2">
            <CircleAlert size={15} className="mt-0.5 shrink-0" /> {turn.error}
          </div>
        )}
      </div>
    </div>
  );
}

export function Chat(props: {
  active: boolean; // on screen; a hidden chat has no size, so it re-measures when it comes back
  examples: string[];
  students: number;
  batches: number;
  asOf: string;
  hasKey: boolean;
  turns: Turn[];
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const { active, examples, students, batches, asOf, hasKey, turns, busy, onSend, onStop } = props;
  const [text, setText] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const pinned = useRef(true);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [turns, active]);
  useEffect(() => {
    const el = input.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text, active]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (!text.trim() || busy || !hasKey) return;
    pinned.current = true;
    onSend(text.trim());
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) submit(e);
  };

  const composer = (
    <form onSubmit={submit}
      className="flex items-end gap-2 rounded-2xl border border-line bg-white px-3 py-2 transition focus-within:border-brand-400 focus-within:ring-4 focus-within:ring-brand-100">
      <textarea ref={input} rows={1} value={text} disabled={!hasKey} onChange={(e) => setText(e.target.value)} onKeyDown={onKey}
        maxLength={4000} aria-label="Ask about a student or batch" placeholder="Ask about a student or a batch"
        className="flex-1 resize-none bg-transparent outline-none text-[15px] leading-6 py-1.5 placeholder:text-faint disabled:cursor-not-allowed" />
      {busy ? (
        <button type="button" onClick={onStop} aria-label="Stop" title="Stop"
          className="size-9 shrink-0 rounded-xl bg-ink text-white grid place-items-center hover:bg-body transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100">
          <Square size={13} fill="currentColor" />
        </button>
      ) : (
        <button type="submit" disabled={!text.trim() || !hasKey} aria-label="Send" title="Send"
          className="size-9 shrink-0 rounded-xl bg-brand-700 text-white grid place-items-center hover:bg-brand-800 transition disabled:bg-line disabled:text-faint outline-none focus-visible:ring-4 focus-visible:ring-brand-100">
          <ArrowUp size={17} />
        </button>
      )}
    </form>
  );

  if (turns.length === 0) {
    return (
      <section className="flex-1 min-w-0 overflow-y-auto">
        <div className="max-w-[680px] mx-auto px-4 sm:px-8 pt-[9vh] pb-16">
          <h1 className="text-[32px] sm:text-[42px] leading-[1.1] font-bold text-ink tracking-tight">Ask about any student</h1>
          <p className="mt-4 text-[16px] text-body leading-relaxed">
            How they use the app, which lectures they finish or skip, how they do in tests, their weak subjects, their rank
            in the batch and any test-integrity reviews. Answers come from the data, with the batch for comparison.
          </p>
          {!hasKey && (
            <div role="alert" className="mt-6 flex items-start gap-2 rounded-lg bg-warn-50 border border-warn-200 text-warn-700 text-[13px] px-3 py-2.5">
              <CircleAlert size={15} className="mt-0.5 shrink-0" />
              The agent needs an Ollama Cloud key. Set OLLAMA_API_KEY in .env (local) or in the host's secrets, then restart the server.
            </div>
          )}
          <div className="mt-7">{composer}</div>
          <div className="mt-4 flex flex-wrap gap-2">
            {examples.map((x) => (
              <button key={x} type="button" disabled={!hasKey || busy} onClick={() => onSend(x)}
                className="rounded-full border border-line bg-white px-3.5 py-1.5 text-[13px] text-body transition hover:border-brand-300 hover:text-ink outline-none focus-visible:ring-4 focus-visible:ring-brand-100 disabled:opacity-50">
                {x}
              </button>
            ))}
          </div>
          <p className="mt-10 text-[13px] text-muted">
            Demo data: {students || 300} made-up students in {batches || 3} batches, as of {asOf ? longDate(asOf) : "the snapshot date"}.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="flex-1 min-h-0 min-w-0 flex flex-col">
      <div ref={scroller} onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }} className="flex-1 overflow-y-auto">
        <div className="max-w-[760px] mx-auto px-4 sm:px-7 py-7 space-y-6">
          {turns.map((t, i) =>
            t.role === "user" ? (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-700 text-white px-4 py-2.5 text-[14.5px] leading-relaxed whitespace-pre-wrap">
                  {t.text}
                </div>
              </div>
            ) : (
              <AssistantTurn key={i} turn={t} live={busy && i === turns.length - 1} />
            ),
          )}
        </div>
      </div>
      <div className="border-t border-line bg-canvas px-4 sm:px-7 py-4">
        <div className="max-w-[760px] mx-auto">{composer}</div>
      </div>
    </section>
  );
}
