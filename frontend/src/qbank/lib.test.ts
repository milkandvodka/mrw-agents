import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeFindings, parseSse, scorecard, toCsv } from "./lib.ts";
import type { Finding, Queue } from "./lib.ts";

const q = { id: "PH-1", question: "Stem?", options: ["a", "b", "c", "d"], answer: 0, explanation: "", topic: "T" };
const f: Finding = { key: "k1", check: "key_conflict", severity: "high", source: "exact", question_id: "PH-1",
  related_ids: ["PH-9"], title: "t", detail: "d", question: q };

test("merge adds new findings as open and keeps existing status", () => {
  const a = mergeFindings({}, [f], 1);
  assert.equal(a.added, 1);
  const confirmed: Queue = { k1: { ...a.queue.k1, status: "confirmed" } };
  const b = mergeFindings(confirmed, [f, { ...f }], 2);
  assert.equal(b.added, 0);
  assert.equal(b.queue.k1.status, "confirmed");
});

test("scorecard counts related ids and real findings", () => {
  const { queue } = mergeFindings({}, [f, { ...f, key: "k2", question_id: "PH-7", related_ids: [] }], 1);
  const s = scorecard(queue, [{ type: "key_conflict", ids: ["PH-9", "PH-3"] }, { type: "blank_option", ids: ["PH-5"] }]);
  assert.equal(s.caught, 1);
  assert.equal(s.total, 2);
  assert.deepEqual(s.byType.key_conflict, [1, 1]);
  assert.equal(s.real, 1);
});

test("csv escapes quotes and neutralises spreadsheet formulas", () => {
  const { queue } = mergeFindings({}, [{ ...f, detail: '=HYPERLINK("x")', title: 'say "hi"' }], 1);
  const csv = toCsv(Object.values(queue));
  assert.ok(csv.startsWith("status,"));
  assert.ok(csv.includes(`"'=HYPERLINK(""x"")"`));
  assert.ok(csv.includes(`"say ""hi"""`));
  assert.ok(csv.includes(`"exact match"`));
});

test("sse parser handles split chunks", () => {
  const a = parseSse('data: {"type":"token","text":"he');
  assert.deepEqual(a.events, []);
  const b = parseSse(a.rest + 'llo"}\n\ndata: {"type":"done"}\n\n');
  assert.deepEqual(b.events, [{ type: "token", text: "hello" }, { type: "done" }]);
  assert.equal(b.rest, "");
});

test("lib has no regular expressions", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(new URL("./lib.ts", import.meta.url), "utf8");
  assert.ok(!src.includes("RegExp") && !src.includes(".test(") && !src.includes("replace(/"));
});
