import { ArrowDown, ArrowUp, ChevronRight, CircleAlert, MessageSquare, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { getStudent, getStudents } from "./api";
import type { Integrity, StudentDetail, StudentRow } from "./api";
import { daysBetween, hm, longDate, shortDate } from "./lib";

const ago = (n: number) => (n === 0 ? "today" : n === 1 ? "yesterday" : `${n} days ago`);
const pct = (x: number | null) => (x == null ? "–" : `${Math.round(100 * x)}%`);

const INTEGRITY: Record<Exclude<Integrity, null>, [string, string]> = {
  confirmed: ["Confirmed", "bg-bad-50 text-bad-700 border-bad-200"],
  open: ["Under review", "bg-warn-50 text-warn-700 border-warn-200"],
  cleared: ["Cleared", "bg-ok-50 text-ok-700 border-ok-200"],
};

function Badge({ v }: { v: Integrity }) {
  if (!v) return <span className="text-faint">–</span>;
  const [label, cls] = INTEGRITY[v];
  return <span className={`inline-block rounded-full border px-2 py-0.5 text-[12px] font-semibold whitespace-nowrap ${cls}`}>{label}</span>;
}

type Col = { key: string; label: string; value: (r: StudentRow) => number | string; cell: (r: StudentRow, asOf: string) => ReactNode };
const COLS: Col[] = [
  { key: "name", label: "Name", value: (r) => r.name, cell: (r) => <span className="font-semibold text-ink">{r.name}</span> },
  { key: "batch", label: "Batch", value: (r) => r.batch, cell: (r) => r.batch },
  { key: "last", label: "Last active", value: (r) => r.last_active ?? "", cell: (r, asOf) => r.last_active
      ? <>{shortDate(r.last_active)} <span className="text-faint">· {ago(daysBetween(r.last_active, asOf))}</span></> : "never" },
  { key: "active", label: "Days active (30)", value: (r) => r.active_30, cell: (r) => r.active_30 },
  { key: "lectures", label: "Lectures watched", value: (r) => r.lectures_full, cell: (r) => <>{r.lectures_full} <span className="text-faint">of 114</span></> },
  { key: "tests", label: "Tests taken", value: (r) => r.tests_taken, cell: (r) => <>{r.tests_taken} <span className="text-faint">of {r.tests_held}</span></> },
  { key: "gt", label: "Latest grand test", value: (r) => (r.latest_gt ? -r.latest_gt.rank / r.latest_gt.of : -2),
    cell: (r) => r.latest_gt ? <>Rank {r.latest_gt.rank} <span className="text-faint">of {r.latest_gt.of}</span></> : <span className="text-faint">–</span> },
  { key: "acc", label: "Accuracy", value: (r) => r.accuracy ?? -1, cell: (r) => pct(r.accuracy) },
  { key: "integrity", label: "Integrity", value: (r) => (r.integrity ? { confirmed: 3, open: 2, cleared: 1 }[r.integrity] : 0), cell: (r) => <Badge v={r.integrity} /> },
];

function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-white">
      <table className="w-full text-[13px]">
        <thead className="bg-canvas text-muted text-left">
          <tr>{head.map((h) => <th key={h} className="px-3 py-2 font-semibold whitespace-nowrap">{h}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="px-3 py-2 whitespace-nowrap">{c}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  );
}

function Detail({ id, asOf, onClose, onAsk }: { id: string; asOf: string; onClose: () => void; onAsk: (text: string) => void }) {
  const [d, setD] = useState<StudentDetail | null>(null);
  const [err, setErr] = useState("");
  const [openSubject, setOpenSubject] = useState("");
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setD(null);
    getStudent(id).then(setD).catch(() => setErr("Couldn't load this student."));
    close.current?.focus();
  }, [id]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-ink/30" onClick={onClose} aria-hidden />
      <aside role="dialog" aria-modal="true" aria-label={d ? d.name : "Student"}
        className="relative h-full w-full max-w-[900px] bg-canvas shadow-2xl overflow-y-auto animate-rise">
        <div className="sticky top-0 z-10 bg-white border-b border-line px-4 sm:px-6 py-3 flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-[18px] font-bold text-ink truncate">{d?.name ?? "Loading…"}</h2>
            {d && <p className="text-[13px] text-muted truncate">{d.batch} · mentor {d.mentor} · joined {longDate(d.joined)}</p>}
          </div>
          {d && (
            <button type="button" className="btn-primary h-9 px-3" onClick={() => onAsk(`How is ${d.name}${d.name === "Priya Sharma" ? ` (${d.batch})` : ""} doing?`)}>
              <MessageSquare size={15} /> <span className="hidden sm:inline">Ask about {d.name.split(" ")[0]}</span><span className="sm:hidden">Ask</span>
            </button>
          )}
          <button ref={close} type="button" className="btn-quiet size-9 px-0" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>
        {err && <div role="alert" className="m-6 flex gap-2 text-bad-700"><CircleAlert size={16} /> {err}</div>}
        {d && (
          <div className="px-4 sm:px-6 py-5 space-y-7">
            <section>
              <h3 className="text-[14px] font-bold text-ink mb-2">App use, last 12 weeks</h3>
              <p className="text-[13px] text-muted mb-2">
                {d.last_active ? `Last active ${longDate(d.last_active)} (${ago(daysBetween(d.last_active, asOf))}).` : "Never opened the app."}
              </p>
              <Table head={["Week", "Days active", "Time"]}
                rows={d.weeks.map((w) => [`${shortDate(w.start)} – ${shortDate(w.end)}`, w.days, hm(w.minutes)])} />
            </section>

            <section>
              <h3 className="text-[14px] font-bold text-ink mb-2">Tests</h3>
              {d.tests.length ? (
                <Table head={["Test", "Date", "Score", "Right / wrong / skipped", "Rank in batch", "Time", "Tab switches"]}
                  rows={d.tests.map((t) => [t.test, shortDate(t.date), `${t.score} / ${t.max}`, `${t.right} / ${t.wrong} / ${t.skipped}`,
                    `${t.rank} of ${t.of}`, `${t.minutes} of ${t.limit} min`, t.tabs])} />
              ) : <p className="text-[13px] text-muted">No tests taken.</p>}
              {d.missed.length > 0 && <p className="mt-2 text-[13px] text-muted">Missed: {d.missed.join(", ")}.</p>}
            </section>

            <section>
              <h3 className="text-[14px] font-bold text-ink mb-2">Subjects</h3>
              <div className="rounded-xl border border-line bg-white divide-y divide-line overflow-hidden">
                <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 px-3 py-2 bg-canvas text-[12.5px] font-semibold text-muted">
                  <span>Subject</span><span className="w-20 text-right">Lectures</span><span className="w-16 text-right">Accuracy</span><span className="w-14 text-right">Batch</span>
                </div>
                {d.subjects.map((s) => {
                  const full = s.lectures.filter((l) => (l.pct ?? 0) >= 90).length;
                  const open = openSubject === s.subject;
                  return (
                    <div key={s.subject}>
                      <button type="button" aria-expanded={open} onClick={() => setOpenSubject(open ? "" : s.subject)}
                        className="w-full grid grid-cols-[1fr_auto_auto_auto] gap-3 px-3 py-2 text-left text-[13px] hover:bg-canvas/70 outline-none focus-visible:bg-canvas">
                        <span className="flex items-center gap-1.5 text-ink"><ChevronRight size={14} className={`text-faint transition ${open ? "rotate-90" : ""}`} />{s.subject}</span>
                        <span className="w-20 text-right">{full} of {s.lectures.length}</span>
                        <span className="w-16 text-right">{pct(s.accuracy)}</span>
                        <span className="w-14 text-right text-faint">{pct(s.batch_accuracy)}</span>
                      </button>
                      {open && (
                        <ul className="px-9 pb-3 space-y-1 text-[12.5px]">
                          {s.lectures.map((l) => (
                            <li key={l.title} className="flex justify-between gap-3">
                              <span className="text-body">{l.title}</span>
                              <span className={l.pct == null ? "text-bad-700" : "text-muted"}>
                                {l.pct == null ? "not watched" : `${l.pct}% watched, first ${shortDate(l.first ?? "")}`}
                              </span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  );
                })}
              </div>
              <p className="mt-2 text-[12.5px] text-muted">Lectures: watched fully (90%+). Accuracy: right answers out of attempted, in every test taken.</p>
            </section>

            <section>
              <h3 className="text-[14px] font-bold text-ink mb-2">Test-integrity reviews</h3>
              {d.reviews.length ? (
                <ul className="space-y-2">
                  {d.reviews.map((r) => (
                    <li key={r.test + r.opened} className="rounded-xl border border-line bg-white p-3 text-[13px]">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-semibold text-ink">{r.test}</span>
                        <Badge v={r.outcome as Integrity} />
                        <span className="text-muted">{r.signal}</span>
                      </div>
                      <p className="mt-1 text-body">{r.note}</p>
                    </li>
                  ))}
                </ul>
              ) : <p className="text-[13px] text-muted">None.</p>}
            </section>
          </div>
        )}
      </aside>
    </div>
  );
}

export function Students({ asOf, onAsk }: { asOf: string; onAsk: (text: string) => void }) {
  const [rows, setRows] = useState<StudentRow[] | null>(null);
  const [err, setErr] = useState("");
  const [query, setQuery] = useState("");
  const [batch, setBatch] = useState("All");
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: "name", desc: false });
  const [selected, setSelected] = useState("");
  useEffect(() => { getStudents().then(setRows).catch(() => setErr("Couldn't load the students.")); }, []);

  const shown = useMemo(() => {
    const col = COLS.find((c) => c.key === sort.key) ?? COLS[0];
    const q = query.trim().toLowerCase();
    return (rows ?? [])
      .filter((r) => (batch === "All" || r.batch === batch) && (!q || r.name.toLowerCase().includes(q)))
      .sort((a, b) => {
        const x = col.value(a), y = col.value(b);
        const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
        return sort.desc ? -c : c;
      });
  }, [rows, query, batch, sort]);

  return (
    <section className="flex-1 min-w-0 overflow-y-auto">
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-wrap items-end gap-3">
          <div className="mr-auto">
            <h1 className="text-[24px] font-bold text-ink">Students</h1>
            <p className="text-[13px] text-muted">The data the agent answers from. Made-up students, as of {longDate(asOf)}. Open one to see everything.</p>
          </div>
          <label className="relative">
            <span className="sr-only">Search by name</span>
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-faint" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by name"
              className="h-9 w-56 rounded-lg border border-line bg-white pl-8 pr-3 text-[13.5px] outline-none focus:border-brand-400 focus:ring-4 focus:ring-brand-100" />
          </label>
          <div role="group" aria-label="Batch" className="flex rounded-lg border border-line bg-white p-0.5">
            {["All", "Batch A", "Batch B", "Batch C"].map((b) => (
              <button key={b} type="button" aria-pressed={batch === b} onClick={() => setBatch(b)}
                className={`h-8 px-3 rounded-md text-[13px] font-semibold transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100 ${batch === b ? "bg-brand-700 text-white" : "text-body hover:bg-canvas"}`}>
                {b.replace("Batch ", "")}
              </button>
            ))}
          </div>
        </div>
        {err && <div role="alert" className="mt-4 text-bad-700 text-[14px]">{err}</div>}
        <div className="mt-4 overflow-x-auto rounded-xl border border-line bg-white">
          <table className="w-full text-[13px]">
            <thead className="bg-canvas text-muted text-left">
              <tr>
                {COLS.map((c) => (
                  <th key={c.key} aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"} className="px-3 py-2 font-semibold whitespace-nowrap">
                    <button type="button" className="inline-flex items-center gap-1 hover:text-ink outline-none focus-visible:text-ink"
                      onClick={() => setSort({ key: c.key, desc: sort.key === c.key ? !sort.desc : c.key !== "name" && c.key !== "batch" })}>
                      {c.label}
                      {sort.key === c.key && (sort.desc ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {!rows && !err && <tr><td colSpan={COLS.length} className="px-3 py-6 text-center text-muted">Loading…</td></tr>}
              {shown.map((r) => (
                <tr key={r.id} onClick={() => setSelected(r.id)} className="cursor-pointer hover:bg-brand-50/40">
                  {COLS.map((c, i) => (
                    <td key={c.key} className="px-3 py-2 whitespace-nowrap">
                      {i === 0 ? (
                        <button type="button" className="text-left outline-none focus-visible:underline" onClick={(e) => { e.stopPropagation(); setSelected(r.id); }}>
                          {c.cell(r, asOf)}
                        </button>
                      ) : c.cell(r, asOf)}
                    </td>
                  ))}
                </tr>
              ))}
              {rows && !shown.length && <tr><td colSpan={COLS.length} className="px-3 py-6 text-center text-muted">No student matches.</td></tr>}
            </tbody>
          </table>
        </div>
        {rows && <p className="mt-2 text-[12.5px] text-muted">{shown.length} of {rows.length} students</p>}
      </div>
      {selected && <Detail id={selected} asOf={asOf} onClose={() => setSelected("")} onAsk={onAsk} />}
    </section>
  );
}
