import { ArrowLeft, ClipboardCheck, GraduationCap } from "lucide-react";
import type { ReactNode } from "react";

export type AgentKey = "qbank" | "mentor";

// Each agent keeps its own colour in the switch, whichever palette the page around it is wearing.
const TEAL = "#0e6f86", INDIGO = "#3a43a0";
const AGENTS = [ // the switch's order, left to right; the first is the one the site opens on
  { key: "mentor", name: "Mentor Desk", short: "Mentor", Icon: GraduationCap, color: INDIGO },
  { key: "qbank", name: "QBank Auditor", short: "QBank", Icon: ClipboardCheck, color: TEAL },
] as const;

export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <rect x="1.5" y="1.5" width="14" height="14" rx="4" fill={TEAL} />
      <rect x="8.5" y="8.5" width="14" height="14" rx="4" fill={INDIGO} stroke="#fff" strokeWidth="2" />
    </svg>
  );
}

export function AgentBar({ agent, onSwitch, onBack, right }: {
  agent: AgentKey;
  onSwitch: (a: AgentKey) => void;
  onBack?: () => void;
  right?: ReactNode;
}) {
  return (
    <header className="h-14 shrink-0 bg-white border-b border-line flex items-center px-4 sm:px-6 lg:grid lg:grid-cols-[1fr_auto_1fr]">
      <div className={`flex items-center gap-2 min-w-0 lg:mr-0 ${onBack ? "mr-2" : "md:mr-4"}`}>
        {onBack && (
          <button type="button" className="btn-quiet h-9 -ml-2 px-2 text-[14px]" onClick={onBack} aria-label="Back">
            <ArrowLeft size={17} /> <span className="hidden sm:inline">Back</span>
          </button>
        )}
        <span className="hidden md:flex items-center gap-2.5 text-[15px] font-bold text-ink tracking-[-0.01em] whitespace-nowrap">
          <Mark /> mrw agents
        </span>
      </div>

      <nav aria-label="Agents" className="flex items-center gap-0.5 rounded-xl border border-line bg-canvas p-[3px]">
        {AGENTS.map(({ key, name, short, Icon, color }) => {
          const on = key === agent;
          return (
            <button key={key} type="button" aria-current={on ? "page" : undefined} onClick={() => on || onSwitch(key)}
              className={`h-7 pl-[3px] pr-2.5 rounded-[9px] flex items-center gap-2 text-[13.5px] font-semibold whitespace-nowrap transition outline-none focus-visible:ring-4 focus-visible:ring-brand-100 ${on
                ? "bg-white text-ink shadow-[0_1px_2px_rgba(16,24,40,.08),0_0_0_1px_var(--color-line)] cursor-default"
                : "text-muted hover:text-ink hover:bg-white/60"}`}>
              <span className="size-[22px] rounded-md grid place-items-center transition-colors" style={on ? { background: color, color: "#fff" } : { color }}>
                <Icon size={14} strokeWidth={2.2} />
              </span>
              <span className="sm:hidden">{short}</span>
              <span className="hidden sm:inline">{name}</span>
            </button>
          );
        })}
      </nav>

      <div className="ml-auto lg:ml-0 lg:justify-self-end flex items-center gap-2">{right}</div>
    </header>
  );
}
