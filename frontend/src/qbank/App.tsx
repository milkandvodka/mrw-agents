import { Check, FlaskConical, X } from "lucide-react";
import { AgentBar } from "../AgentBar";
import type { AgentKey } from "../AgentBar";
import { useCallback, useEffect, useRef, useState } from "react";
import { getOverview, resetSession, streamChat } from "./api";
import type { AgentEvent, Overview } from "./api";
import { Chat } from "./Chat";
import type { Part, ToolPart, Turn } from "./Chat";
import { PLANT_NAMES, mergeFindings, scorecard } from "./lib";
import type { Planted, Queue, Status } from "./lib";
import { QueuePanel } from "./Queue";

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full or blocked: the session still works, it just won't survive a refresh */
  }
}
const newId = () => crypto.randomUUID().replaceAll("-", "");

// a turn that was mid-flight when the page closed can't resume
function settle(turns: Turn[]): Turn[] {
  return turns.map((t) =>
    t.role === "assistant" && t.parts.some((p) => p.kind === "tool" && p.status === "running")
      ? { ...t, parts: t.parts.map((p) => (p.kind === "tool" && p.status === "running" ? { ...p, status: "error" as const } : p)), error: t.error ?? "Interrupted when the page was closed." }
      : t,
  );
}

function SelfTest({ queue, planted }: { queue: Queue; planted: Planted[] }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const s = scorecard(queue, planted);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);
  const r = 9, c = 2 * Math.PI * r;
  return (
    <div ref={box} className="relative">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="h-9 pl-2 pr-3 rounded-lg border border-line bg-white hover:bg-canvas flex items-center gap-2 whitespace-nowrap text-[13px] font-semibold text-ink transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100">
        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden className="-rotate-90">
          <circle cx="12" cy="12" r={r} fill="none" stroke="var(--color-line)" strokeWidth="3" />
          <circle cx="12" cy="12" r={r} fill="none" stroke="var(--color-ok-700)" strokeWidth="3" strokeLinecap="round"
            strokeDasharray={c} strokeDashoffset={c * (1 - s.caught / Math.max(1, s.total))} className="transition-all duration-700" />
        </svg>
        <span className="tabular-nums">{s.caught}/{s.total}</span> <span className="hidden sm:inline text-muted font-medium">planted errors found</span>
      </button>
      {open && (
        <div className="absolute right-0 top-11 z-20 w-[340px] rounded-xl border border-line bg-white shadow-[0_18px_50px_-20px_rgba(15,42,48,.35)] p-4 animate-rise">
          <div className="flex items-center gap-2 text-ink font-semibold"><FlaskConical size={16} className="text-brand-700" /> Planted-error self-test</div>
          <p className="mt-1.5 text-[12.5px] text-muted leading-relaxed">
            Before launch, {s.total} errors were planted in the bank at random. The agent isn't told which questions they are.
          </p>
          <ul className="mt-3 space-y-1.5">
            {Object.entries(s.byType).map(([type, [hit, tot]]) => (
              <li key={type} className="flex items-center gap-2 text-[13px]">
                <span className={`size-4 rounded-full grid place-items-center ${hit === tot ? "bg-ok-50 text-ok-700" : "bg-canvas text-faint"}`}>
                  {hit === tot ? <Check size={11} strokeWidth={3} /> : <X size={10} strokeWidth={3} />}
                </span>
                <span className="flex-1 text-body">{PLANT_NAMES[type] ?? type}</span>
                <span className="tabular-nums text-muted">{hit} of {tot}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[12px] text-muted leading-relaxed">
            Five are exact copies, repeats or blanks that the full audit finds. The other five sit in the Autonomic nervous system topic, and only the AI review can catch them: two keys changed while their explanations still give the original answer, and three silent wrong keys with no explanation.
          </p>
          <div className="mt-3 rounded-lg bg-canvas px-3 py-2 text-[12.5px] text-body">
            <span className="font-semibold tabular-nums">{s.real}</span> findings are on original, unplanted questions: problems in the public data itself.
          </div>
          <p className="mt-3 text-[11px] text-faint">Independent prototype on public MedMCQA data.</p>
        </div>
      )}
    </div>
  );
}

export default function App({ active, onSwitch }: { active: boolean; onSwitch: (a: AgentKey) => void }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [loadError, setLoadError] = useState("");
  const [sessionId, setSessionId] = useState(() => load("qa.session", "") || newId());
  const [turns, setTurns] = useState<Turn[]>(() => settle(load<Turn[]>("qa.turns", [])));
  const [queue, setQueueState] = useState<Queue>(() => load<Queue>("qa.queue", {}));
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<{ text: string; undo?: () => void } | null>(null);
  const queueRef = useRef(queue);
  const abort = useRef<AbortController | null>(null);
  const activeRef = useRef(active); // only the agent on screen reacts to the browser's Back
  useEffect(() => { activeRef.current = active; }, [active]);

  const setQueue = (q: Queue) => {
    queueRef.current = q;
    setQueueState(q);
  };

  useEffect(() => {
    getOverview().then(setOverview).catch(() => setLoadError("Can't reach the auditor server. Is it running?"));
  }, []);
  useEffect(() => save("qa.session", sessionId), [sessionId]);
  useEffect(() => save("qa.turns", turns), [turns]);
  useEffect(() => save("qa.queue", queue), [queue]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4500);
    return () => clearTimeout(t);
  }, [toast]);

  const updateLast = (fn: (t: Extract<Turn, { role: "assistant" }>) => void) =>
    setTurns((ts) => {
      const copy = ts.slice();
      const last = copy[copy.length - 1];
      if (!last || last.role !== "assistant") return ts;
      const next = { ...last, parts: last.parts.map((p) => ({ ...p })) as Part[] };
      fn(next);
      copy[copy.length - 1] = next;
      return copy;
    });

  const onEvent = (e: AgentEvent) => {
    if (e.type === "token") {
      updateLast((t) => {
        const p = t.parts[t.parts.length - 1];
        if (p?.kind === "text") p.text += e.text;
        else t.parts.push({ kind: "text", text: e.text });
      });
    } else if (e.type === "tool_start") {
      updateLast((t) => t.parts.push({ kind: "tool", id: e.id, name: e.name, args: e.args, status: "running", summary: "", content: "", added: 0 }));
    } else if (e.type === "progress") {
      updateLast((t) => {
        const p = t.parts.find((x): x is ToolPart => x.kind === "tool" && x.id === e.id);
        if (p) p.progress = { done: e.done, total: e.total };
      });
    } else if (e.type === "tool_end") {
      const r = mergeFindings(queueRef.current, e.findings, Date.now());
      setQueue(r.queue);
      updateLast((t) => {
        const p = t.parts.find((x): x is ToolPart => x.kind === "tool" && x.id === e.id);
        if (p) Object.assign(p, { status: "done", summary: e.summary, content: e.content, added: r.added });
      });
    } else if (e.type === "error") {
      updateLast((t) => {
        t.error = e.message;
        t.parts.forEach((p) => { if (p.kind === "tool" && p.status === "running") p.status = "error"; });
      });
    }
  };

  const send = async (text: string) => {
    if (busy) return;
    setTurns((ts) => [...ts, { role: "user", text }, { role: "assistant", parts: [] }]);
    setBusy(true);
    const ac = new AbortController();
    abort.current = ac;
    try {
      await streamChat(sessionId, text, onEvent, ac.signal);
    } catch {
      onEvent({ type: "error", message: ac.signal.aborted
        ? "Stopped. Anything already found stays in the queue."
        : "Lost the connection to the server. Anything already found stays in the queue; try again." });
    } finally {
      setBusy(false);
      abort.current = null;
    }
  };

  const setStatus = useCallback((key: string, status: Status) => {
    const prev = queueRef.current[key];
    if (!prev) return;
    setQueue({ ...queueRef.current, [key]: { ...prev, status } });
    const words = { confirmed: "Issue confirmed", dismissed: "Dismissed", open: "Moved back to open" }[status];
    setToast({ text: `${words}: ${prev.question_id}`, undo: () => setQueue({ ...queueRef.current, [key]: { ...queueRef.current[key], status: prev.status } }) });
  }, []);

  const started = turns.length > 0 || Object.keys(queue).length > 0;

  // home = a fresh start: stop the agent, forget this chat and its queue
  const goHome = useRef(() => {});
  goHome.current = () => {
    if (!started) return;
    abort.current?.abort();
    resetSession(sessionId);
    setSessionId(newId());
    setTurns([]);
    setQueue({});
  };
  // leaving home adds a history entry, so the browser's back button comes home too
  useEffect(() => {
    if (started && history.state?.view !== "work") history.pushState({ view: "work" }, "");
  }, [started]);
  useEffect(() => {
    const onPop = () => { if (activeRef.current) goHome.current(); };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const back = () => (history.state?.view === "work" ? history.back() : goHome.current());

  return (
    <div className="h-full flex flex-col">
      <AgentBar agent="qbank" onSwitch={onSwitch} onBack={started ? back : undefined}
        right={overview && Object.values(queue).some((f) => !f.question_id.startsWith("NEW-")) && <SelfTest queue={queue} planted={overview.planted} />} />

      {loadError ? (
        <div role="alert" className="m-6 rounded-xl border border-bad-200 bg-bad-50 text-bad-700 p-4 text-[14px]">{loadError}</div>
      ) : (
        <main className={`animate-fade flex-1 min-h-0 ${started ? "overflow-y-auto lg:overflow-hidden lg:grid lg:grid-cols-[minmax(420px,5fr)_7fr]" : "flex"}`}>
          <Chat
            active={active}
            suggestions={overview?.suggestions ?? []}
            examples={overview?.examples ?? []}
            questions={overview?.questions ?? 0}
            hasKey={overview?.has_key ?? true}
            turns={turns}
            busy={busy}
            onSend={send}
            onStop={() => abort.current?.abort()}
          />
          {started && <QueuePanel queue={queue} onStatus={setStatus} active={active} />}
        </main>
      )}

      {toast && (
        <div role="status" className="fixed bottom-5 left-1/2 -translate-x-1/2 z-30 flex items-center gap-3 rounded-xl bg-ink text-white text-[13px] pl-4 pr-2 py-2 shadow-xl animate-rise">
          {toast.text}
          {toast.undo && (
            <button type="button" className="rounded-md px-2 py-1 font-semibold text-brand-300 hover:bg-white/10"
              onClick={() => { toast.undo?.(); setToast(null); }}>
              Undo
            </button>
          )}
        </div>
      )}
    </div>
  );
}
