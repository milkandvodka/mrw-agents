import { Check, ChevronDown, Download, Sparkles, Undo2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { LETTERS, toCsv } from "./lib";
import type { Item, Queue, Status } from "./lib";

const SEVERITY = {
  high: { label: "High priority", bar: "bg-bad-700" },
  medium: { label: "Medium priority", bar: "bg-warn-700" },
  low: { label: "Low priority", bar: "bg-faint" },
};
const RANK = { high: 0, medium: 1, low: 2 };
const VIEWS: { id: Status; label: string }[] = [
  { id: "open", label: "Open" },
  { id: "confirmed", label: "Confirmed" },
  { id: "dismissed", label: "Dismissed" },
];
const PAGE = 30;

function Card({ f, open, onToggle, onStatus }: {
  f: Item;
  open: boolean;
  onToggle: () => void;
  onStatus: (key: string, s: Status) => void;
}) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [open]);
  const q = f.question;
  const sev = SEVERITY[f.severity];
  const marks = new Map<number, ["key" | "alt", string]>([[q.answer, ["key", "Key"]]]);
  if (f.suggested != null && f.suggested !== q.answer)
    marks.set(f.suggested, ["alt", f.check === "key_conflict" ? "Key in the other copy"
      : f.check === "ai_explanation_conflict" ? "Explanation backs" : "AI suggests"]);
  const fresh = Date.now() - f.addedAt < 4000;

  return (
    <article
      ref={ref}
      className={`relative rounded-xl bg-white border transition ${open ? "border-brand-400 ring-4 ring-brand-100" : "border-line hover:border-[#cfdcdf]"} ${fresh ? "animate-rise" : ""}`}
    >
      <span className={`absolute left-0 top-3 bottom-3 w-[3px] rounded-r-full ${sev.bar}`} aria-hidden />
      <button type="button" onClick={onToggle} aria-expanded={open} className="w-full text-left rounded-xl py-3 pl-5 pr-3.5 outline-none focus-visible:bg-canvas/60">
        <span className="flex items-center gap-2.5">
          <span className="sr-only">{sev.label}: </span>
          <span className="flex-1 min-w-0 truncate font-semibold text-ink text-[14.5px]">{f.title}</span>
          {f.source !== "exact" && (
            <span className="shrink-0 inline-flex items-center gap-1 rounded-full bg-warn-50 text-warn-700 text-[11.5px] font-semibold px-2 py-0.5">
              <Sparkles size={12} /> AI opinion
            </span>
          )}
          <ChevronDown size={16} className={`shrink-0 text-faint transition ${open ? "rotate-180" : ""}`} />
        </span>
        <span className={`mt-1 text-[13px] text-muted leading-relaxed ${open ? "block" : "line-clamp-1"}`}>{f.reason || f.detail}</span>
      </button>

      {open && (
        <div className="pb-4 pl-5 pr-3.5">
          <div className="rounded-lg bg-canvas p-3">
            <p className="text-[14px] font-semibold text-ink leading-snug">{q.question}</p>
            <ul className="mt-2.5 space-y-1.5">
              {q.options.map((o, i) => {
                const m = marks.get(i);
                const row = m?.[0] === "key" ? "bg-ok-50 border-ok-200" : m?.[0] === "alt" ? "bg-bad-50 border-bad-200" : "bg-white border-line";
                const dot = m?.[0] === "key" ? "bg-ok-700 border-ok-700 text-white" : m?.[0] === "alt" ? "border-bad-700 text-bad-700" : "border-line text-muted";
                return (
                  <li key={i} className={`flex items-center gap-2.5 rounded-lg border px-3 py-2 text-[13.5px] ${row}`}>
                    <span className={`size-5 shrink-0 rounded-full border grid place-items-center text-[10.5px] font-bold ${dot}`}>{LETTERS[i]}</span>
                    <span className="flex-1 min-w-0 text-body">{o || <em className="text-faint">(blank)</em>}</span>
                    {m && <span className={`shrink-0 text-[11px] font-semibold ${m[0] === "key" ? "text-ok-700" : "text-bad-700"}`}>{m[1]}</span>}
                  </li>
                );
              })}
            </ul>
            {f.check === "ai_explanation_conflict" && q.explanation && (
              <p className="mt-2.5 text-[12.5px] text-muted leading-relaxed">
                <span className="font-semibold text-body">Explanation: </span>
                {q.explanation.length > 260 ? q.explanation.slice(0, 259) + "…" : q.explanation}
              </p>
            )}
          </div>

          <div className="mt-3 flex items-center gap-2">
            {f.status === "open" ? (
              <>
                <button type="button" className="btn-primary" onClick={() => onStatus(f.key, "confirmed")}><Check size={15} /> Confirm issue</button>
                <button type="button" className="btn-ghost" onClick={() => onStatus(f.key, "dismissed")}><X size={15} /> Dismiss</button>
              </>
            ) : (
              <>
                <span className={`text-[12.5px] font-semibold ${f.status === "confirmed" ? "text-ok-700" : "text-muted"}`}>
                  {f.status === "confirmed" ? "Confirmed by faculty" : "Dismissed"}
                </span>
                <button type="button" className="btn-quiet ml-auto" onClick={() => onStatus(f.key, "open")}><Undo2 size={14} /> Move back to open</button>
              </>
            )}
          </div>
        </div>
      )}
    </article>
  );
}

export function QueuePanel({ queue, onStatus, active }: { queue: Queue; onStatus: (key: string, s: Status) => void; active: boolean }) {
  const [view, setView] = useState<Status>("open");
  const [limit, setLimit] = useState(PAGE);
  const [sel, setSel] = useState<string | null>(null);

  const all = useMemo(() => Object.values(queue), [queue]);
  const count = (s: Status) => all.filter((f) => f.status === s).length;
  const items = useMemo(
    () => all
      .filter((f) => f.status === view)
      .sort((a, b) => RANK[a.severity] - RANK[b.severity] || a.question_id.localeCompare(b.question_id)),
    [all, view],
  );
  const shown = items.slice(0, limit);

  // keyboard triage: J/K move, C confirm, D dismiss, U reopen
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("textarea, input, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey || !shown.length || !active) return; // keys typed on the other agent aren't for this queue
      const i = Math.max(0, shown.findIndex((f) => f.key === sel));
      const k = e.key.toLowerCase();
      if (k === "j" || e.key === "ArrowDown") setSel(shown[Math.min(i + (sel ? 1 : 0), shown.length - 1)].key);
      else if (k === "k" || e.key === "ArrowUp") setSel(shown[Math.max(i - 1, 0)].key);
      else if (sel && ((k === "c" && view === "open") || (k === "d" && view === "open") || (k === "u" && view !== "open"))) {
        onStatus(sel, k === "c" ? "confirmed" : k === "d" ? "dismissed" : "open");
        setSel(shown[i + 1]?.key ?? shown[i - 1]?.key ?? null);
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shown, sel, view, onStatus, active]);

  const exportCsv = () => {
    const url = URL.createObjectURL(new Blob([toCsv(all)], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "qbank-review-queue.csv" });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="min-h-[85vh] lg:min-h-0 flex flex-col">
      <div className="px-5 sm:px-7 pt-6 pb-3">
        <div className="flex items-center gap-3">
          <h2 className="text-[19px] font-bold text-ink tracking-tight">Review queue</h2>
          {all.length > 0 && (
            <button type="button" className="btn-ghost ml-auto" onClick={exportCsv}>
              <Download size={14} /> Export CSV
            </button>
          )}
        </div>
        {all.length > 0 && (
          <>
            <p className="mt-0.5 text-[13px] text-muted">Open one to see the question. Confirm or dismiss it.</p>
            <div role="tablist" aria-label="Status" className="mt-4 inline-flex rounded-lg bg-white border border-line p-0.5">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  role="tab"
                  aria-selected={view === v.id}
                  type="button"
                  onClick={() => { setView(v.id); setSel(null); setLimit(PAGE); }}
                  className={`h-7 px-3 rounded-md text-[12.5px] font-semibold transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100 ${view === v.id ? "bg-brand-700 text-white" : "text-muted hover:text-ink"}`}
                >
                  {v.label} <span className={`tabular-nums ${view === v.id ? "text-brand-100" : "text-faint"}`}>{count(v.id)}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto px-5 sm:px-7 pb-6 space-y-2">
        {all.length === 0 ? (
          <p className="mt-16 mx-auto max-w-[34ch] text-center text-[14px] text-muted leading-relaxed">
            Problems the agent finds will show up here for you to confirm or dismiss.
          </p>
        ) : items.length === 0 ? (
          <p className="mt-8 text-center text-[13.5px] text-muted">Nothing {view} yet.</p>
        ) : (
          shown.map((f) => (
            <Card key={f.key} f={f} open={sel === f.key} onToggle={() => setSel(sel === f.key ? null : f.key)} onStatus={onStatus} />
          ))
        )}
        {items.length > limit && (
          <button type="button" className="btn-ghost w-full h-10" onClick={() => setLimit(limit + PAGE)}>
            Show {Math.min(PAGE, items.length - limit)} more of {items.length - limit}
          </button>
        )}
      </div>
    </section>
  );
}
