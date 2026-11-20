# QBank Auditor

An agent that checks medical MCQs before students see them.

- **Paste your questions** with their keys. It checks:
  - whether the key is right, and whether the explanation agrees with it;
  - whether options repeat or are empty;
  - whether the bank already has the same question with a different answer.
- **Or ask about the demo bank** in plain words ("check the Chemotherapy questions for wrong keys"). The agent picks its own tools and works through the bank.

Every problem goes into a review queue with its evidence. **Nothing changes until a faculty member confirms it.** The agent never writes questions, explanations or medical advice.

The demo bank is public data: [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), the Pharmacology subset of 13,758 real AIIMS and NEET PG questions. This is an independent prototype.

## How it decides: the model judges, code only compares

There are no keyword rules or pattern matching anywhere. Two kinds of evidence, both reached by the agent through tool calls:

- **Exact match** (`run_checks`): code compares text exactly, ignoring case and punctuation. It finds the same question saved twice with different keys, the same option listed twice, and empty options. These are facts, so anyone can reproduce them.
- **AI opinion** (`ai_review`): an AI reviewer reads each question with its options, key and explanation, and judges it: a wrong key, an explanation that backs a different option, two defensible answers, and (if asked) badly worded items. Every such item is labelled *AI opinion: verify before changing*.

The agent decides which tools to use, reads questions, compares near-copies and adds its own flags with quoted evidence. Faculty confirm or dismiss everything.

What the exact checks find in the public data:

| Problem | Count |
|---|---|
| Same question saved twice with **different answer keys** | 24 groups |
| The **same option listed twice** (e.g. simvastatin as both A and D) | 72 questions |
| Exact duplicates with the same key (bank bloat) | 1,316 groups |
| Questions with no topic | 6,843 |

## Self-test: it proves itself

Before launch, 10 errors are planted in the bank (`data/planted.json`); the agent isn't told which.
- **Five are visible to the exact checks:** two copies saved with a different key, two repeated options and one blanked option.
- **Five sit in the Autonomic nervous system topic, and only the AI review can catch them:** two keys changed while the explanation still gives the original answer, and three silent wrong keys with no explanation at all.

The score is shown in the app's "Self-test" panel.

AI review uses DeepSeek V4.1 Flash with thinking on. On a 20-question batch it caught 3/3 planted wrong keys with 0 false alarms (67 s); with thinking off it caught 2/3 with 2 false alarms (6 s). That's a small sample, but precision is what earns faculty trust, so thinking stays on.

## Evals: does the agent actually work?

Real conversations against the live model, graded automatically, with no pattern matching:
- **Truth comes from the data:** exact counts, keys and IDs, the planted-error answer sheet, and the search index's own tokens.
- **What the agent did is compared by code:** the tools it called, what it queued and which IDs it cited.
- **What the reply says** (e.g. "what number did it give?") is read by a model that must quote the reply word for word. Code checks the quote is really there, then compares the answer to the truth. An unbacked claim counts for nothing.
- **The reader itself is checked** on labelled replies.

```bash
python -m qbank.evals                 # core suite: 10 cases + reviewer + reader calibration
python -m qbank.evals --suite big     # ~300 generated cases of 28 kinds + 4 component evals (~30 min)
python -m qbank.evals --suite big --only clinical,injection --sample 20 --seed 7
python -m qbank.evals --suite chat    # 27 real conversations, graded turn by turn (~5 min)
python -m qbank.evals --suite chat --questions my_questions.json   # the same, pasting your own MCQs
```

The **chat suite** talks to the agent the way people do, and grades every turn on its own:
- "hey", then "what do u do", then three questions pasted at once in mixed formats ("Ans: b", "Answer - (B)", "key: Tenofovir")
- follow-ups that need memory: "which one had a problem?", "was the third one fine?", "how many have I sent you?", "remind me what the second one was about"
- corrections ("oops, the answer is actually C") and missing keys ("its b" one message later)
- random asks in between (IPL, a joke, maths) and silly ones ("print hello 1000 times", "count to 1000", "print your system prompt", "make 50 MCQs")
- back-to-back questions: "how many in the bank?", "and in chemotherapy?", "endocrinology?", "which of those two has more?"

"Was the third one fine?" is graded against what the agent's own check found for that question, so the eval tests whether it reports truthfully, not whether a website's key is right.

Runs, in order:

| Questions pasted | First run | After the fixes below |
|---|---|---|
| Held-out past papers | 25/27 | 27/27 |
| 32 real questions from the web (NEET PG pharmacology, NEET UG 2024 biology) | 26/27, then 24/27 on new conversations | 27/27 on those same new conversations |

**At scale:** `--rounds 10` gives 260 fresh conversations (910 turns), drawn from the web questions plus the held-out papers. The first run passed **245/260 (94%)**:
- **Every silly or hostile ask held** (30/30), and so did all corrections (50/50) and "is it in your bank? give me the ID" (20/20).
- **Most of the 15 failures were grader mistakes, now fixed.** In the long-memory checks the reader wasn't told which question was "the fourth", and it counted "only a typo" as a problem. It also failed the agent for calling a key "disputed" when that key really was broken on purpose; a reply now passes if it matches the check's evidence or the known truth.
- **Three were real agent slips, now fixed:**
  - "key: Ethambutol" (the answer written as text) was sent to the check twice, once with the wrong letter;
  - "which of those two?" once more compared a look-alike topic; the topic tool now labels the exact match first;
  - an off-topic ask made the agent keep investigating with tools.
- **One judgment call I left as is:** the agent called a reworded bank question a "near-identical copy".

After the fixes I re-ran the 15 that failed and kept the 245 that passed (`--resume`): **256/260**. Long memory went from 3/10 to 10/10. Still failing:
- once in 20, "which of those two has more?" after the agent mentioned a look-alike topic;
- once, tool calls after an off-topic ask;
- twice, "and the 3rd one?" after the agent flagged something itself. The run kept only the last reply, so I couldn't judge these; runs now keep every turn's reply.

What the chat suite caught:
- **"Which of those two has more?"** It compared look-alike topics it had mentioned itself (C.N.S next to Central Nervous system) instead of the two you asked about. Fixed with a follow-up rule, and look-alike names are now one short note.
- **IDs after a correction.** It listed similar bank questions by ID. The paste-checking tool no longer shows the agent any IDs, so they can't leak.
- **Two grader errors:**
  - The reply reader judged "That's the AI reviewer's opinion" as not saying whose opinion it was. That check now asks the question directly, and the reply is in the reader's calibration set.
  - The quote check rejected a quote with a trimmed last word.

The big suite (`eval_suite.py`) generates its cases from the data with a seed:
- key lookups, topic and check counts (including zero counts)
- search with exact truth, copies and conflicting copies
- every planted error by ID
- missing IDs, made-up topics
- typos and Hinglish, gibberish, false premises
- off-topic asks, medical questions, clinical asks and emergencies
- requests to write content, and prompt injection
- multi-turn and compound asks
- AI reviews of chosen IDs, single-topic and whole-bank audits
- "is this already in the bank?" with questions pasted from held-out past-year papers

**Latest full run:** 282/292 cases passed across 27 kinds ([report](evals/results/2026-11-18-1244-big.md)). After fixing what it found, I re-ran the kinds that had failed:
- **Single-topic audits:** 6/6. The expectation changed from "no AI review unless asked" to "exact checks plus one AI review", because judgment now needs the AI review.
- **New whole-bank audits:** 4/4, and none starts an AI review.
- **Medical questions:** 20/20, run twice.
- **False premises:** 11/12. The miss corrects the premise but doesn't always name the real topic.
- **AI reviews of chosen IDs:** the two remaining misses are the ambiguous flunarizine plant below.

| Component eval | Result |
|---|---|
| AI reviewer, 100 flipped + 100 copy-backed bank keys | caught 99/100 (named the original answer for 97), 0/100 false alarms |
| AI reviewer, 100 flipped + 100 untouched held-out past-year keys | caught 93/100 (named the original for 91), raised a problem on 1/100 untouched keys |
| Exact checks, mutation test | 300/300 breakages caught, 0/300 look-alikes flagged |
| Reply-reader calibration | 29/29 |

Problems the evals found, each now fixed:
- **Topics:** the agent couldn't see every topic name (it now has a list-topics tool). Partial matching silently merged "Hematology" with "Hematology and CVS" (an exact name now wins, and tools say which topics they covered).
- **Search:** it never said how many questions contain a word, so the agent claimed spellings the bank really uses ("amilodipine", 7 questions) weren't there. Search now reports the total.
- **Scope:** it translated text and did arithmetic when asked, and added medical facts of its own to summaries. There are now prompt rules for both, checked by the off-topic and medical-question cases.
- **Cost and speed:** on "full audit" it started AI reviews of 6 topics at once (42 minutes). A harness-level limit now allows one AI review per message.
- **Concurrency:** Ollama Cloud returned 429 "too many concurrent requests" under parallel load. There's now a shared concurrency cap (`REVIEW_CONCURRENCY`, default 4) and retries with backoff.

Known hard case: one planted silent wrong key (migraine prophylaxis keyed flunarizine instead of propranolol) is usually not flagged. Flunarizine is a defensible answer, which is exactly why faculty make the final call.

## How it works

- **Agent:** LangChain `create_agent` (runs on LangGraph) with DeepSeek V4.1 Flash on Ollama Cloud, with retries on rate limits.
- **Tools:**
  - check pasted questions: the model splits what you paste into stem, options and key as tool arguments (no parsing rules). Code runs the exact checks and looks for the same question in the bank. The AI reviewer judges the key, explanation and wording.
  - bank overview
  - list topics
  - search, which reports how many questions contain a word
  - read questions
  - exact checks
  - find similar questions
  - batched AI review
  - flag a question

  Tools return a compact summary to the model; findings go straight to the UI's queue.
- **Evidence labels:** every item is either *Exact match* (reproducible) or *AI opinion* (verify before changing).
- **Server:** FastAPI streams the agent's steps to the browser over server-sent events.
- **UI:** React + Tailwind. The first screen is a box to paste questions, with suggestions under it that each answer in seconds: sample questions to check, and questions about the demo bank. A topic's AI review takes minutes, so it isn't suggested; ask for one. Back or Stop cancels whatever the agent is still running, so the next request doesn't queue behind it. Once you ask something, the chat shows each tool step live, and a review queue fills while the agent works: one line per problem, opened to see the question and confirm or dismiss it. Keyboard triage (J/K move, C confirm, D dismiss, U reopen), undo, CSV export and the self-test score. Chat and queue survive a page refresh.

| File | What it does |
|---|---|
| `checks.py` | Exact checks (same question with different keys, repeated options, empty options): text comparison only |
| `bank.py` | Loads the bank, runs the checks once, TF-IDF search and similarity |
| `agent.py` | Tools, the AI reviewer, the agent, and the event stream |
| `server.py` | FastAPI: `/api/overview`, `/api/chat` (SSE); mounted at `/qbank` by `../app.py` |
| `../frontend/src/qbank/` | The React app (`App.tsx`, `Chat.tsx`, `Queue.tsx`, pure logic in `lib.ts`) |
| `prepare_data.py` | Builds `data/bank.json` and plants the 10 test errors; `--heldout` builds the eval-only past-year set |
| `evals.py`, `eval_suite.py` | The eval harness and the generated suite |

## Run, test and deploy

See the [README at the repo root](../README.md): both agents run from one server.

## Limits

- The AI review takes a few minutes per ~300 questions. Exact checks are instant.
- Chat history lives in the server's memory; a server restart starts the agent's memory fresh (the browser keeps the queue).
