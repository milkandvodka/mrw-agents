import { useState } from "react";
import type { AgentKey } from "./AgentBar";
import MentorDesk from "./mentor/App";
import QBankAuditor from "./qbank/App";

const fromHash = (): AgentKey => (location.hash === "#qbank" ? "qbank" : "mentor"); // Mentor Desk is the default

// Both agents stay mounted, so an answer still streaming in one is waiting when you switch back.
export default function Shell() {
  const [agent, setAgent] = useState<AgentKey>(fromHash);
  const pick = (a: AgentKey) => {
    history.replaceState(history.state, "", `#${a}`); // a link to this agent, without a history entry: Back stays each agent's own
    setAgent(a);
  };
  return (
    <>
      <div className="h-full" hidden={agent !== "qbank"}>
        <QBankAuditor active={agent === "qbank"} onSwitch={pick} />
      </div>
      <div className="h-full theme-mentor bg-canvas" hidden={agent !== "mentor"}>
        <MentorDesk active={agent === "mentor"} onSwitch={pick} />
      </div>
    </>
  );
}
