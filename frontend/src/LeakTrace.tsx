import { ExternalLink } from "lucide-react";
import { useEffect, useState } from "react";
import { AgentBar } from "./AgentBar";
import type { AgentKey } from "./AgentBar";

// LeakTrace runs as its own service; this tab shows it in a frame. Its address comes from the server's settings,
// and the frame only loads once the tab is first opened.
export default function LeakTrace({ active, onSwitch }: { active: boolean; onSwitch: (a: AgentKey) => void }) {
  const [url, setUrl] = useState<string | null>(null);
  const [opened, setOpened] = useState(active);
  useEffect(() => { if (active) setOpened(true); }, [active]);
  useEffect(() => {
    fetch("/api/config").then((r) => r.json()).then((c) => setUrl(c.leaktrace_url || "")).catch(() => setUrl(""));
  }, []);
  return (
    <div className="h-full flex flex-col">
      <AgentBar agent="leaktrace" onSwitch={onSwitch}
        right={url ? <a className="btn-quiet h-9 px-2 text-[13.5px]" href={url} target="_blank" rel="noreferrer"><ExternalLink size={15} /> <span className="hidden sm:inline">Open in a new tab</span></a> : undefined} />
      {url === null ? null : url ? (
        opened && <iframe title="LeakTrace" src={url} className="flex-1 w-full border-0 bg-white" allow="fullscreen" />
      ) : (
        <p className="m-auto max-w-[420px] p-6 text-center text-[14px] text-muted">LeakTrace isn't connected on this deployment.</p>
      )}
    </div>
  );
}
