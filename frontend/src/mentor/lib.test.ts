import assert from "node:assert/strict";
import { test } from "node:test";
import { daysBetween, hm, longDate, parseSse, shortDate } from "./lib.ts";

test("hm and dates read like the agent's answers", () => {
  assert.equal(hm(0), "0 min");
  assert.equal(hm(45), "45 min");
  assert.equal(hm(125), "2h 5m");
  assert.equal(shortDate("2026-10-23"), "23 Oct");
  assert.equal(daysBetween("2026-10-23", "2026-11-15"), 23);
});

test("parseSse returns complete events and keeps the unfinished tail", () => {
  const { events, rest } = parseSse('data: {"type":"token","text":"Hi"}\n\ndata: {"type":"do');
  assert.deepEqual(events, [{ type: "token", text: "Hi" }]);
  assert.equal(rest, 'data: {"type":"do');
});

test("longDate formats the snapshot date without shifting it", () => {
  assert.equal(longDate("2026-11-15"), "15 Nov 2026");
  assert.equal(longDate("2026-01-05"), "5 Jan 2026");
  assert.equal(longDate("not a date"), "not a date");
});
