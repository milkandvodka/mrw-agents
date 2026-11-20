import { Users } from "lucide-react";
import { AgentBar } from "../AgentBar";
import type { AgentKey } from "../AgentBar";
import { useEffect, useRef, useState } from "react";
import { getOverview, resetSession, streamChat } from "./api";
import type { AgentEvent, Overview } from "./api";
import { Chat } from "./Chat";
import type { Part, ToolPart, Turn } from "./Chat";
import { Students } from "./Students";

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
    /* storage full or blocked: the chat still works, it just won't survive a refresh */
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

export default function App({ active, onSwitch }: { active: boolean; onSwitch: (a: AgentKey) => void }) {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [loadError, setLoadError] = useState("");
  const [sessionId, setSessionId] = useState(() => load("md.session", newId()));
  const [turns, setTurns] = useState<Turn[]>(() => settle(load<Turn[]>("md.turns", [])));
  const [busy, setBusy] = useState(false);
  const [students, setStudents] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const activeRef = useRef(active); // only the agent on screen reacts to the browser's Back
  useEffect(() => { activeRef.current = active; }, [active]);

  useEffect(() => {
    getOverview().then(setOverview).catch(() => setLoadError("Can't reach the Mentor Desk server. Is it running?"));
  }, []);
  useEffect(() => save("md.session", sessionId), [sessionId]);
  useEffect(() => save("md.turns", turns), [turns]);

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
      updateLast((t) => t.parts.push({ kind: "tool", id: e.id, name: e.name, args: e.args, status: "running", content: "" }));
    } else if (e.type === "tool_end") {
      updateLast((t) => {
        const p = t.parts.find((x): x is ToolPart => x.kind === "tool" && x.id === e.id);
        if (p) Object.assign(p, { status: "done", content: e.content });
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
      onEvent({ type: "error", message: ac.signal.aborted ? "Stopped." : "Lost the connection to the server. Try again." });
    } finally {
      setBusy(false);
      abort.current = null;
    }
  };

  const started = turns.length > 0;
  // home = a fresh start: stop the agent and forget this chat
  const goHome = useRef(() => {});
  goHome.current = () => {
    setStudents(false);
    if (!started) return;
    abort.current?.abort();
    resetSession(sessionId);
    setSessionId(newId());
    setTurns([]);
  };
  // leaving home adds a history entry, so the browser's back button comes home too
  useEffect(() => {
    if (started && history.state?.view !== "chat") history.pushState({ view: "chat" }, "");
  }, [started]);
  useEffect(() => {
    const onPop = () => {
      if (!activeRef.current) return;
      const view = history.state?.view;
      setStudents(view === "students");
      if (view !== "chat" && view !== "students") goHome.current();
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const back = () => {
    if (students) {
      if (history.state?.view === "students") history.back();
      else setStudents(false);
    } else if (history.state?.view === "chat") history.back();
    else goHome.current();
  };
  const openStudents = () => {
    if (students) return;
    history.pushState({ view: "students" }, "");
    setStudents(true);
  };
  // "Ask about this student": the students page becomes the chat in history, so Back leaves the chat as usual
  const askAbout = (text: string) => {
    history.replaceState({ view: "chat" }, "");
    setStudents(false);
    send(text);
  };

  return (
    <div className="h-full flex flex-col">
      <AgentBar agent="mentor" onSwitch={onSwitch} onBack={started || students ? back : undefined}
        right={<button type="button" onClick={openStudents} aria-pressed={students}
          className={`btn h-9 px-3 text-[14px] border ${students ? "bg-brand-50 border-brand-300 text-brand-800" : "bg-white border-line text-body hover:bg-canvas"}`}>
          <Users size={16} /> <span className="hidden min-[400px]:inline">Students</span>
        </button>} />
      {loadError ? (
        <div role="alert" className="m-6 rounded-xl border border-bad-200 bg-bad-50 text-bad-700 p-4 text-[14px]">{loadError}</div>
      ) : (
        <main className="animate-fade flex-1 min-h-0 flex">
          {students ? <Students asOf={overview?.as_of ?? ""} onAsk={askAbout} /> : <Chat
            active={active}
            examples={overview?.examples ?? []}
            students={overview?.students ?? 0}
            batches={overview?.batches.length ?? 0}
            asOf={overview?.as_of ?? ""}
            hasKey={overview?.has_key ?? true}
            turns={turns}
            busy={busy}
            onSend={send}
            onStop={() => abort.current?.abort()}
          />}
        </main>
      )}
    </div>
  );
}
