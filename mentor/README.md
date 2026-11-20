# Mentor Desk

Mentors ask about their students in plain words, and an agent answers from the data:

- how a student uses the app;
- which lectures they finish, skip or watch out of order;
- how they do in tests, which subjects and question kinds they get wrong, and their rank in the batch;
- any test-integrity reviews.

It also answers questions about a whole batch: who has gone quiet, who is climbing or sliding, who topped a test.

> "Who in Batch A has gone quiet lately?" · "How is Ishita Sharma doing?" · "Has anyone cheated in Grand Test 3?"

The **Students** button shows the data itself: every student with last active, lectures watched, tests taken, latest grand test rank, accuracy and integrity status, searchable and sortable. Open one to see their weekly app use, every test, every subject with each lecture watched or not, and their reviews, or ask the agent about them.

## How it decides: the model judges, code only counts

- **Tools return facts.** Ten tools read a SQLite database and return exact numbers, each next to the batch figure: a student report, app use by week, lecture history, test results, accuracy by subject and topic, integrity evidence, rankings, and a read-only SQL tool for anything else. No regex, no keyword rules, no thresholds that label a student.
- **The model connects them.** A weak topic whose lecture was never watched; a falling rank next to falling app time; a 21-minute finish with a near-top score.
- **Integrity is evidence, not a verdict.** Fast finishes, tab switches and alike answer sheets are signals. Only a review marked "confirmed" means a student cheated, and the agent says so plainly when a review cleared someone.
- **Read-only at the engine.** The database opens read-only, and SQLite's authorizer refuses anything but reads (writes, ATTACH, PRAGMA), so the SQL tool can't change data however the query is written.
- **Names.** Students are found by name. Two students with the same name make the agent ask which one; for a typo, the tool lists the closest names and the agent says which one it used.

## The data: a made-up cohort with planted stories

There is no real student data here. `generate.py` builds a NEET PG cohort from one seed, so it's the same every time. Each name is drawn within one community (North, South, Bengali, Marathi/Gujarati, Muslim, Christian, Sikh), with no famous people:

- 300 students in three batches;
- 19 subjects and 114 lectures;
- 19 subject tests and 5 grand tests (about 170,000 answers);
- daily app use and integrity reviews.

The data is snapshotted on 15 Nov 2026.

Ability, engagement and preparation drive the answers. For example, a student who watched a topic's lecture does better on that topic's questions.

Fifteen stories are planted on purpose and listed in `data/stories.json`:

- **Activity:** a dropout (gone for 23 days), a decliner (rank and app time both sliding), an improver (batch rank 67 → 4), and a new joiner with no tests yet.
- **Lectures:** a Pharmacology skipper (no Pharmacology lectures, 6% accuracy there against the batch's 55%, 58% overall), a student who watches a quarter of each lecture, one who watches every subject last-to-first, and one who watches everything but never takes a test.
- **Performance:** a top student with one weak subject.
- **Integrity:** a copier (93% of wrong answers matching another student's against a typical 42%, confirmed), the student they copied from (cleared), an open speed flag, and a cleared tab-switch flag.
- **Names:** two students named Priya Sharma.

## Evals

`python -m mentor.evals` runs the agent live on cases whose truth comes from the database, never from the model:

- **Stories:** every planted story.
- **Counts:** exact counts.
- **Behaviour:** greetings, off-topic and silly asks, a missing student, data it doesn't have, edit requests, and a check-in draft.
- **Ethics:** no guessing religion or gender from names, no insulting messages, no deleting data.
- **Custom queries:** questions only the SQL tool can answer.
- **Follow-ups:** questions that need memory.

`--rounds N` adds N rounds of generated questions about random students and tests: scores, ranks, toppers, lectures, last active, tests missed, typos and Hinglish.

`--chat N` adds N rounds of whole conversations, about 12 per round:
- **A full session:** hello, what can you do, a student, their tests, a silly ask, their weakest subject, bye.
- **Memory:** two students, then "which missed more?" and "who did I ask about first?".
- **Interruptions:** scores, then an off-topic ask, then "who scored higher?".
- **Corrections:** "sorry, I meant…".
- **Several questions at once.**
- **Back-to-back counts.**
- **Prompt injection.**
- **The two Priya Sharmas.**

A model reads each reply and must quote it, so a claim only counts if the reply really makes it. Every answer about a student is also checked for he/she, because the data has no gender.

| Run | Result |
|---|---|
| Fixed cases, first run (earlier data, before names were redrawn) | 28/29: the one miss was "he" for a student; fixed by a note in every per-student tool result |
| Fixed + 3 generated rounds (earlier data) | 56/56 |
| Fixed + 20 generated rounds, seed 11 (earlier data) | 209/209 |
| Full suite, seed 21: 36 fixed + 108 generated + 180 conversations | 315/324. "Which one has more?" compared all three batches instead of the two asked about (4); the SQL tool took "watched fully" as 100%; he/she in 22 of 655 replies |
| After a follow-up example, "fully" = 90%+ in SQL, the pronoun note naming the student | 318/324, then 314/324 on an identical rerun: he/she slips vary from 2 to 7 of 655 replies run to run |
| After tool sentences say "they", and a different first name is treated as a different student | **320/324**: 2 he/she slips in 655 replies, 1 playful pirate decline, 1 reader misread (the agent was right) |

Reports: `evals/results/`.

## Run, test and deploy

See the [README at the repo root](../README.md): both agents run from one server.

## Limits

- The data is synthetic. A platform's real data would replace `generate.py`; the tools only need the same tables.
- Chat history lives in the server's memory.
- There are no mentor logins: every mentor sees every batch.
