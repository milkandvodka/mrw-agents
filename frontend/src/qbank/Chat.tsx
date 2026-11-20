import {
  ArrowUp, BookOpen, Check, ChevronRight, CircleAlert, ClipboardCheck, Flag, GitCompareArrows, Layers, ListChecks, LoaderCircle,
  Search, ShieldCheck, Sparkles, Square, X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Suggestion } from "./api";

export type ToolPart = {
  kind: "tool";
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  summary: string;
  content: string;
  added: number;
  progress?: { done: number; total: number };
};
export type Part = { kind: "text"; text: string } | ToolPart;
export type Turn = { role: "user"; text: string } | { role: "assistant"; parts: Part[]; error?: string };

const TOOL_LABELS: Record<string, string> = {
  check_questions: "Checking your questions",
  bank_overview: "Reading the bank overview",
  list_topics: "Listing topics",
  search_questions: "Searching questions",
  get_questions: "Reading questions",
  run_checks: "Running quality checks",
  find_similar: "Looking for similar questions",
  ai_review: "AI reviewing questions",
  flag_question: "Adding to the review queue",
};
const TOOL_ICONS: Record<string, LucideIcon> = {
  check_questions: ListChecks,
  bank_overview: Layers,
  search_questions: Search,
  get_questions: BookOpen,
  run_checks: ShieldCheck,
  find_similar: GitCompareArrows,
  ai_review: Sparkles,
  flag_question: Flag,
};

function describe(args: Record<string, unknown>): string {
  const bits: string[] = [];
  if (args.query) bits.push(`“${args.query}”`);
  if (args.topic) bits.push(`topic ${args.topic}`);
  if (args.check && args.check !== "all") bits.push(String(args.check).replaceAll("_", " "));
  if (Array.isArray(args.ids)) bits.push(`${args.ids.length} questions`);
  if (Array.isArray(args.questions)) bits.push(`${args.questions.length} pasted question${args.questions.length === 1 ? "" : "s"}`);
  if (args.question_id) bits.push(String(args.question_id));
  return bits.length ? ` (${bits.join(", ")})` : "";
}

function label(p: ToolPart): string {
  const base = TOOL_LABELS[p.name] ?? p.name;
  if (p.status === "running") return base + describe(p.args) + "…";
  if (p.status === "error") return base + describe(p.args);
  const t = p.summary.trim();
  const s = (t.endsWith(":") ? t.slice(0, -1) : t).trim();
  if (p.name === "get_questions") return `Read ${Array.isArray(p.args.ids) ? p.args.ids.length : ""} questions`;
  if (p.name === "find_similar") return `Looked for questions similar to ${p.args.question_id ?? "the given text"}`;
  if (p.name === "search_questions") return `Searched${describe(p.args)}: ${s}`;
  return s || base;
}

function ToolStep({ p }: { p: ToolPart }) {
  const [open, setOpen] = useState(false);
  const Icon = TOOL_ICONS[p.name] ?? Layers;
  const pct = p.progress ? Math.round((100 * p.progress.done) / Math.max(1, p.progress.total)) : 0;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left hover:bg-canvas/70 outline-none focus-visible:bg-canvas"
      >
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
        <span className="min-w-0 flex-1">
          <span className="block text-[13.5px] text-ink truncate">{label(p)}</span>
          {p.progress && p.status === "running" && (
            <span className="mt-1.5 flex items-center gap-2">
              <span className="h-1 flex-1 rounded-full bg-line overflow-hidden">
                <span className="block h-full bg-brand-500 transition-all duration-500" style={{ width: `${pct}%` }} />
              </span>
              <span className="text-[11px] text-muted tabular-nums">{p.progress.done} of {p.progress.total}</span>
            </span>
          )}
        </span>
        {p.added > 0 && (
          <span className="shrink-0 text-[11.5px] font-semibold text-brand-700 bg-brand-50 rounded-full px-2 py-0.5 tabular-nums">
            +{p.added} to queue
          </span>
        )}
        {p.content && <ChevronRight size={15} className={`shrink-0 text-faint transition ${open ? "rotate-90" : ""}`} />}
      </button>
      {open && p.content && (
        <pre className="mx-3.5 mb-3 max-h-64 overflow-auto rounded-lg bg-canvas border border-line p-3 text-[12px] leading-relaxed text-body whitespace-pre-wrap font-sans">
          {p.content}
        </pre>
      )}
    </div>
  );
}

function AssistantTurn({ turn, live }: { turn: Extract<Turn, { role: "assistant" }>; live: boolean }) {
  // group consecutive tool calls into one step list
  const blocks: (Part | ToolPart[])[] = [];
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
        <ClipboardCheck size={15} />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {blocks.map((b, i) =>
          Array.isArray(b) ? (
            <div key={i} className="rounded-xl border border-line bg-white divide-y divide-line overflow-hidden">
              {b.map((p) => <ToolStep key={p.id} p={p} />)}
            </div>
          ) : b.kind === "text" ? (
            <div key={i} className="md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{b.text}</ReactMarkdown>
            </div>
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
  suggestions: Suggestion[];
  examples: string[];
  questions: number;
  hasKey: boolean;
  turns: Turn[];
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const { active, suggestions, examples, questions, hasKey, turns, busy, onSend, onStop } = props;
  const [text, setText] = useState("");
  const [nextExample, setNextExample] = useState(() => Math.floor(Math.random() * 100));  // a different one first each visit
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
    <div className="flex items-end gap-2 rounded-2xl border border-line bg-canvas/70 px-3 py-2 transition focus-within:border-brand-400 focus-within:bg-white focus-within:ring-4 focus-within:ring-brand-100">
      <textarea
        ref={input}
        id="ask"
        rows={1}
        value={text}
        disabled={!hasKey}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        maxLength={4000}
        aria-label="Message the auditor"
        placeholder="Paste questions to check, or ask about the demo bank"
        className="flex-1 resize-none bg-transparent outline-none text-[14.5px] leading-6 py-1.5 placeholder:text-faint disabled:cursor-not-allowed"
      />
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
    </div>
  );

  if (turns.length === 0) {
    return (
      <section className="flex-1 overflow-y-auto bg-white">
        <div className="max-w-[680px] mx-auto px-5 sm:px-8 pt-[7vh] pb-16">
          <h1 className="text-[32px] sm:text-[40px] leading-[1.12] font-bold text-ink tracking-tight">
            Check your MCQs before students see them
          </h1>
          <p className="mt-4 text-[16px] text-body leading-relaxed">
            Paste questions with their answer keys. An AI agent checks whether each key is right, whether the explanation
            agrees with it, whether options repeat, and whether the bank already has the question with a different answer.
            You make the final call.
          </p>
          {!hasKey && (
            <div role="alert" className="mt-6 flex items-start gap-2 rounded-lg bg-warn-50 border border-warn-200 text-warn-700 text-[13px] px-3 py-2.5">
              <CircleAlert size={15} className="mt-0.5 shrink-0" />
              The agent needs an Ollama Cloud key. Set OLLAMA_API_KEY in .env (local) or in the host's secrets, then restart the server.
            </div>
          )}
          <form onSubmit={submit} className="mt-7">
            <label htmlFor="paste" className="sr-only">Questions to check</label>
            <textarea
              id="paste"
              rows={5}
              value={text}
              disabled={!hasKey}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e); }}
              maxLength={4000}
              placeholder={"Paste one or more questions, for example:\nWhich drug is ...?\nA. ...   B. ...   C. ...   D. ...\nAnswer: B"}
              className="block w-full resize-y rounded-2xl border border-line bg-canvas/70 px-4 py-3 text-[14.5px] leading-6 text-ink outline-none transition placeholder:text-faint focus:border-brand-400 focus:bg-white focus:ring-4 focus:ring-brand-100 disabled:cursor-not-allowed"
            />
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button type="submit" disabled={!text.trim() || !hasKey || busy}
                className="inline-flex items-center gap-2 h-11 px-5 rounded-xl bg-brand-700 text-white text-[15px] font-semibold hover:bg-brand-800 transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100 disabled:opacity-50 disabled:pointer-events-none">
                <ListChecks size={18} /> Check questions
              </button>
              {examples.length > 0 && (!text.trim() || examples.includes(text)) && (
                <button type="button" className="btn-ghost h-11 px-4 rounded-xl text-[14px]" disabled={!hasKey}
                  onClick={() => { setText(examples[nextExample % examples.length]); setNextExample(nextExample + 1); }}>
                  {text.trim() ? "Another example" : "Use an example"}
                </button>
              )}
            </div>
          </form>

          <div className="mt-6 flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button key={s.label} type="button" disabled={!hasKey || busy} onClick={() => onSend(s.prompt)}
                className="rounded-full border border-line bg-white px-3.5 py-1.5 text-left text-[13px] text-body transition hover:border-brand-300 hover:text-ink outline-none focus-visible:ring-4 focus-visible:ring-brand-100 disabled:opacity-50">
                {s.label}
              </button>
            ))}
          </div>
          <p className="mt-10 text-[13px] text-muted">
            Demo bank: {questions ? questions.toLocaleString() : "thousands of"} real pharmacology questions from past NEET PG
            and AIIMS papers (public MedMCQA data).
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="h-[85vh] lg:h-auto min-h-0 flex flex-col bg-white lg:border-r border-line">
      <div
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget;
          pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        className="flex-1 overflow-y-auto"
      >
        <div className="max-w-[720px] mx-auto px-5 sm:px-7 py-7 space-y-6">
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

      <form onSubmit={submit} className="border-t border-line bg-white px-5 sm:px-7 py-4">
        <div className="max-w-[720px] mx-auto">{composer}</div>
      </form>
    </section>
  );
}
