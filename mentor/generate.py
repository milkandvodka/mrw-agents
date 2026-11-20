"""Build data/mentor.db: a made-up NEET PG cohort (students, lectures, tests, answers, daily app use, test-integrity reviews)
and data/stories.json, the answer sheet of student stories planted on purpose (a dropout, a lecture skipper, a copier...).

Everything comes from one seed, so the same database is built every time. Run:  python -m mentor.generate
"""
import json
import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
DB = ROOT / "data" / "mentor.db"
SEED = 2027
AS_OF = date(2026, 11, 15)  # the data snapshot: "today" for every answer the agent gives
N_STUDENTS = 300

BATCHES = [("Batch A", "Dr. Kavya Iyer", date(2026, 1, 5)), ("Batch B", "Dr. Rohan Mehta", date(2026, 4, 6)),
           ("Batch C", "Dr. Sana Qureshi", date(2026, 7, 6))]
COURSE = {  # subject -> lectures, in the order the course teaches them
    "Anatomy": ["Upper Limb", "Lower Limb", "Thorax", "Abdomen and Pelvis", "Head and Neck", "Neuroanatomy"],
    "Physiology": ["Nerve and Muscle", "Cardiovascular Physiology", "Respiratory Physiology", "Renal Physiology",
                   "Endocrine Physiology", "Neurophysiology"],
    "Biochemistry": ["Carbohydrate Metabolism", "Lipid Metabolism", "Proteins and Amino Acids", "Enzymes", "Vitamins",
                     "Molecular Biology"],
    "Pathology": ["Cell Injury and Adaptation", "Inflammation and Repair", "Neoplasia", "Haematology",
                  "Cardiovascular Pathology", "Renal Pathology"],
    "Pharmacology": ["General Pharmacology", "Autonomic Nervous System", "Cardiovascular Drugs", "CNS Drugs",
                     "Antimicrobials", "Anticancer Drugs"],
    "Microbiology": ["General Bacteriology", "Immunology", "Systemic Bacteriology", "Virology", "Mycology", "Parasitology"],
    "Forensic Medicine": ["Identification", "Thanatology", "Injuries", "Forensic Toxicology", "Sexual Offences",
                          "Medical Jurisprudence"],
    "Community Medicine": ["Epidemiology", "Biostatistics", "Communicable Diseases", "Nutrition", "Health Programmes",
                           "Environment and Health"],
    "ENT": ["Ear and Hearing", "External and Middle Ear", "Inner Ear and Vertigo", "Nose and Sinuses", "Pharynx", "Larynx"],
    "Ophthalmology": ["Optics and Refraction", "Cornea", "Lens and Cataract", "Glaucoma", "Retina", "Neuro-ophthalmology"],
    "Medicine": ["Cardiology", "Respiratory Medicine", "Gastroenterology", "Nephrology", "Neurology", "Endocrinology"],
    "Surgery": ["Trauma and Burns", "Breast", "Thyroid and Parathyroid", "Hepatobiliary Surgery", "Urology", "Vascular Surgery"],
    "Obstetrics and Gynaecology": ["Normal Pregnancy", "Complications of Pregnancy", "Labour", "Menstrual Disorders",
                                   "Gynaecological Oncology", "Infertility"],
    "Paediatrics": ["Growth and Development", "Neonatology", "Nutrition and Immunisation", "Paediatric Infections",
                    "Genetic Disorders", "Paediatric Cardiology"],
    "Orthopaedics": ["Upper Limb Fractures", "Lower Limb Fractures", "Bone Tumours", "Bone Infections", "Spine",
                     "Paediatric Orthopaedics"],
    "Dermatology": ["Papulosquamous Disorders", "Vesiculobullous Disorders", "Skin Infections", "Leprosy",
                    "Sexually Transmitted Infections", "Pigmentary Disorders"],
    "Psychiatry": ["Schizophrenia", "Mood Disorders", "Anxiety Disorders", "Substance Use", "Personality Disorders",
                   "Child Psychiatry"],
    "Anaesthesia": ["Airway Management", "Inhalational Agents", "Intravenous Agents", "Muscle Relaxants",
                    "Local Anaesthetics", "Regional Anaesthesia"],
    "Radiology": ["Radiation Physics", "Chest Imaging", "Neuroimaging", "Abdominal Imaging", "Musculoskeletal Imaging",
                  "Contrast Agents"],
}
GRAND_TESTS = [date(2026, 4, 12), date(2026, 6, 14), date(2026, 8, 9), date(2026, 9, 27), date(2026, 11, 8)]
NAMES = {  # community -> (share of students, first names, surnames): a name is drawn within one community
    "North": (0.33, ["Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Ayush", "Diya", "Gauri", "Harsh", "Ishaan", "Ishita",
                     "Kunal", "Manav", "Mohit", "Naina", "Neha", "Nikhil", "Pooja", "Pranav", "Priya", "Rahul", "Riya",
                     "Rohit", "Saanvi", "Sahil", "Shreya", "Siddharth", "Sneha", "Tanvi", "Tarun", "Varun", "Yash",
                     "Abhinav", "Ankita", "Deepika", "Gaurav", "Kavya", "Ritika", "Shivani", "Vivek", "Aditya"],
              ["Sharma", "Verma", "Gupta", "Agarwal", "Mishra", "Pandey", "Saxena", "Tiwari", "Sinha", "Yadav", "Chauhan",
               "Thakur", "Kapoor", "Malhotra", "Srivastava", "Dubey", "Chaudhary", "Rastogi"]),
    "South": (0.2, ["Karthik", "Lakshmi", "Vidya", "Arjun", "Meera", "Divya", "Sanjay", "Deepa", "Harini", "Aravind",
                    "Keerthana", "Anjali", "Suresh", "Nandini", "Rohan", "Gayathri", "Vishnu", "Swathi", "Abhishek", "Ramya",
                    "Srinivas", "Pavithra", "Rahul", "Sneha"],
              ["Iyer", "Nair", "Reddy", "Menon", "Rao", "Pillai", "Shetty", "Bhat", "Hegde", "Naidu", "Krishnan",
               "Subramanian", "Gowda", "Kurup"]),
    "East": (0.12, ["Arnab", "Debjani", "Ishita", "Rohan", "Sayan", "Tanushree", "Riya", "Soham", "Aritra", "Payel",
                    "Sourav", "Moumita", "Anirban", "Priyanka", "Rupam", "Ananya"],
             ["Mukherjee", "Chatterjee", "Bose", "Banerjee", "Ghosh", "Das", "Sen", "Dutta", "Roy"]),
    "West": (0.15, ["Aditi", "Omkar", "Siddhi", "Tejas", "Gauri", "Pranav", "Sakshi", "Hardik", "Krupa", "Mihir", "Neha",
                    "Rutuja", "Yash", "Dhruv", "Janhvi", "Parth", "Isha", "Nikhil"],
             ["Kulkarni", "Desai", "Patel", "Joshi", "Deshpande", "Patil", "Shah", "Mehta", "Gokhale", "Pawar"]),
    "Muslim": (0.12, ["Aisha", "Farhan", "Imran", "Rehan", "Zoya", "Hina", "Sana", "Sameer", "Ayaan", "Zainab", "Faiz",
                      "Mehreen", "Arif", "Nida", "Saad", "Alina"],
               ["Khan", "Ansari", "Siddiqui", "Qureshi", "Shaikh", "Hussain", "Mirza", "Syed"]),
    "Christian": (0.04, ["Ann", "Joel", "Rhea", "Neil", "Sharon", "Kevin", "Merin", "Allen", "Tanya", "Ryan"],
                  ["Fernandes", "D'Souza", "Pereira", "Thomas", "George", "Mathew", "Joseph"]),
    "Sikh": (0.04, ["Harpreet", "Gurpreet", "Jaspreet", "Manpreet", "Simran", "Navjot", "Amandeep", "Kiranjit"],
             ["Gill", "Sandhu", "Sidhu", "Dhillon", "Grewal", "Bedi"]),
}
FAMOUS = {"Rohit Sharma", "Arjun Kapoor", "Kunal Kapoor", "Siddharth Malhotra", "Ananya Pandey", "Arjun Reddy", "Imran Khan",
          "Hina Khan", "Sana Khan", "Meera Nair", "Anjali Menon", "Riya Sen", "Tanushree Dutta", "Ishita Dutta", "Hardik Patel",
          "Navjot Sidhu", "Neha Joshi", "Vivek Agarwal"}  # real people a reader would recognise
KINDS = [("Recall", 0.5), ("Clinical vignette", 0.35), ("Image-based", 0.15)]

# The planted stories, by student id. Each is something a mentor would want to catch; evals check the agent finds them.
STORIES = {
    "S017": ("dropout", "Was active almost daily, then stopped completely 23 days ago; missed Grand Test 5."),
    "S042": ("lecture skipper", "Never watched a Pharmacology lecture and scores far below the batch in Pharmacology."),
    "S063": ("partial watcher", "Starts most lectures but watches only about a quarter of each."),
    "S088": ("out of order", "Watches each subject's lectures last-to-first."),
    "S105": ("copier", "Copied S131's answers in Grand Test 3; the review confirmed it."),
    "S131": ("copy source", "Their answers were copied in Grand Test 3; the review cleared them."),
    "S150": ("open speed flag", "Finished Grand Test 5 in 21 of 120 minutes with a near-top score and 17 tab switches; review open."),
    "S177": ("cleared flag", "Flagged for tab switches in Grand Test 2; cleared (power cut, rejoined from phone)."),
    "S199": ("improver", "Grand test rank has climbed steadily from the bottom third to the top."),
    "S222": ("decliner", "Grand test rank and daily app time have both fallen steadily."),
    "S240": ("name twin", "Shares the name Priya Sharma with S261 (another batch)."),
    "S261": ("name twin", "Shares the name Priya Sharma with S240 (another batch)."),
    "S288": ("new joiner", "Joined 4 days ago: two lectures, no tests yet."),
    "S011": ("one weak subject", "Among the batch's best overall, but weak in Microbiology."),
    "S275": ("never tests", "Watches nearly every lecture in full but has never taken a test."),
}


def _d(x: date) -> str:
    return x.isoformat()


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def build(seed: int = SEED) -> tuple[dict[str, list[tuple]], list[dict]]:
    rng = random.Random(seed)
    t: dict[str, list[tuple]] = {k: [] for k in ("batches", "students", "lectures", "watch", "activity", "tests", "questions",
                                                 "attempts", "answers", "integrity_reviews")}
    t["batches"] = [(n, m, _d(s)) for n, m, s in BATCHES]

    lectures, n = [], 0  # (id, subject, title, course_order, minutes)
    for subject, titles in COURSE.items():
        for title in titles:
            n += 1
            lectures.append((f"L{n:03d}", subject, title, n, rng.randrange(45, 125, 5)))
    t["lectures"] = lectures
    subjects = list(COURSE)

    # tests: one per subject (held as the course reaches it) and five grand tests over everything
    span = (AS_OF - timedelta(days=12) - date(2026, 2, 8)).days
    tests = [(f"{subject} Test", "Subject test", subject, _d(date(2026, 2, 8) + timedelta(days=round(i * span / (len(subjects) - 1)))), 30, 45)
             for i, subject in enumerate(subjects)]
    tests += [(f"Grand Test {i}", "Grand test", None, _d(day), 100, 120) for i, day in enumerate(GRAND_TESTS, 1)]
    tests = [(f"T{i:02d}", *x) for i, x in enumerate(sorted(tests, key=lambda x: x[3]), 1)]
    t["tests"] = tests

    questions, qn = [], 0  # (id, test_id, number, subject, topic, kind, difficulty, answer); difficulty value kept aside
    diff = {}
    for tid, _name, kind, subject, _day, nq, _mins in tests:
        for k in range(1, nq + 1):
            qn += 1
            s = subject or rng.choice(subjects)
            b = rng.gauss(0, 0.9)
            qid = f"Q{qn:04d}"
            diff[qid] = (b, rng.randrange(4))  # the distractor most wrong answers go to
            questions.append((qid, tid, k, s, rng.choice(COURSE[s]), rng.choices([k_ for k_, _ in KINDS], [w for _, w in KINDS])[0],
                              "Easy" if b < -0.5 else "Hard" if b > 0.5 else "Medium", rng.randrange(4)))
    t["questions"] = questions
    by_test = {}
    for q in questions:
        by_test.setdefault(q[1], []).append(q)

    # students
    names, groups = set(), list(NAMES.values())
    while len(names) < N_STUDENTS:  # first name and surname from the same community; Priya Sharma is kept for the twins
        _w, first, last = rng.choices(groups, [g[0] for g in groups])[0]
        name = f"{rng.choice(first)} {rng.choice(last)}"
        if name != "Priya Sharma" and name not in FAMOUS:
            names.add(name)
    names = sorted(names)
    rng.shuffle(names)
    students = []
    for i in range(N_STUDENTS):
        sid = f"S{i + 1:03d}"
        bname, _m, start = BATCHES[i % 3]
        joined = start + timedelta(days=int(rng.expovariate(1 / 12)))
        students.append({"id": sid, "name": names[i], "batch": bname, "joined": min(joined, AS_OF - timedelta(days=30)),
                         "ability": rng.gauss(0, 1), "engagement": rng.betavariate(2.5, 1.8),
                         "affinity": {s: rng.gauss(0, 0.35) for s in subjects}, "trend": 0.0, "stopped": None,
                         "watch_style": "normal", "tests": True, "fade": 0.0})
    S = {s["id"]: s for s in students}

    def move(sid, batch):  # join a batch near its start, so the story has time to happen
        start = next(st for n, _m, st in BATCHES if n == batch)
        S[sid].update(batch=batch, joined=start + timedelta(days=rng.randrange(0, 6)))

    # plant the stories
    for sid in ("S017", "S042", "S063", "S088", "S150", "S177", "S199", "S222", "S011"):
        move(sid, "Batch A")
    for sid in ("S105", "S131", "S261", "S288"):
        move(sid, "Batch C")
    move("S240", "Batch B")
    S["S017"].update(engagement=0.92, ability=0.6, stopped=AS_OF - timedelta(days=23))
    S["S042"]["affinity"]["Pharmacology"] = -2.2
    S["S042"].update(ability=0.3, engagement=0.7, watch_style="skip Pharmacology")
    S["S063"].update(engagement=0.75, watch_style="partial")
    S["S088"].update(engagement=0.75, watch_style="reverse")
    S["S105"].update(ability=-0.8, engagement=0.6)
    S["S131"].update(ability=1.4, engagement=0.8)
    S["S150"].update(ability=-0.2, engagement=0.6)
    S["S177"].update(engagement=0.7)
    S["S199"].update(ability=-0.4, trend=4.0, engagement=0.85)
    S["S222"].update(ability=1.0, trend=-2.6, engagement=0.85, fade=0.85)
    S["S240"]["name"] = S["S261"]["name"] = "Priya Sharma"
    S["S288"].update(joined=AS_OF - timedelta(days=4), engagement=0.8)
    S["S011"].update(ability=2.3, engagement=0.95)
    S["S011"]["affinity"]["Microbiology"] = -2.2
    S["S275"].update(engagement=0.95, tests=False, watch_style="full")
    t["students"] = [(s["id"], s["name"], s["batch"], _d(s["joined"])) for s in students]

    for s in students:
        last_day = s["stopped"] or AS_OF
        enrolled = (AS_OF - s["joined"]).days
        days = [s["joined"] + timedelta(days=k) for k in range((last_day - s["joined"]).days + 1)]

        # daily app use
        mine = []
        for day in days:
            frac = (day - s["joined"]).days / max(enrolled, 1)
            e = s["engagement"] * (1 - s["fade"] * max(0.0, frac - 0.55) / 0.45)
            if rng.random() < 0.1 + 0.85 * e * (0.8 if day.weekday() == 6 else 1):
                mine.append((s["id"], _d(day), max(5, int(rng.lognormvariate(math.log(20 + 110 * e), 0.45)))))
        t["activity"] += mine
        active = {a[1] for a in mine}

        # lectures, roughly in course order, as many as their pace allows
        pace = 0.25 + 0.55 * s["engagement"]  # lectures per active day... spread over the enrolment
        n_watch = 2 if s["id"] == "S288" else min(len(lectures), round(len(active) * pace / 2.4))
        if s["watch_style"] == "full":
            n_watch = len(lectures) - 3
        order = list(lectures)
        if s["watch_style"] == "reverse":
            order = [lec for subj in subjects for lec in reversed([x for x in lectures if x[1] == subj])]
        chosen = []
        for lec in order:
            if len(chosen) >= n_watch:
                break
            if s["watch_style"] == "skip Pharmacology" and lec[1] == "Pharmacology":
                continue
            if s["watch_style"] not in ("full", "reverse") and rng.random() < 0.06:  # the odd lecture skipped
                continue
            chosen.append(lec)
        act_days = sorted(active) or [_d(s["joined"])]
        for k, lec in enumerate(chosen):
            first = act_days[min(len(act_days) - 1, round(k * (len(act_days) - 1) / max(len(chosen), 1)))]
            last = act_days[min(len(act_days) - 1, act_days.index(first) + rng.choice([0, 0, 0, 1, 3, 9]))]
            if s["watch_style"] == "partial":
                pct = rng.randint(10, 40)
            elif s["watch_style"] == "full":
                pct = rng.randint(92, 100)
            else:
                pct = 100 if rng.random() < 0.35 + 0.55 * s["engagement"] else rng.randint(15, 95)
            t["watch"].append((s["id"], lec[0], first, last, pct))
        watched = {w[1]: (w[2], w[4]) for w in t["watch"][-len(chosen):]} if chosen else {}
        topic_lec = {(lec[1], lec[2]): lec[0] for lec in lectures}

        # tests
        if not s["tests"]:
            continue
        for tid, name, kind, subject, day_s, nq, mins in tests:
            day = date.fromisoformat(day_s)
            if day < s["joined"] + timedelta(days=3) or day > last_day:
                continue
            if s["id"] not in ("S105", "S131", "S150", "S177", "S199", "S222") and rng.random() > 0.3 + 0.65 * s["engagement"]:
                continue
            frac = (day - s["joined"]).days / max(enrolled, 1)
            skill = s["ability"] + s["trend"] * (frac - 0.5)
            correct = wrong = skipped = 0
            rows = []
            for qid, _t, _k, subj, topic, _kind, _dl, ans in by_test[tid]:
                b, popular = diff[qid]
                w = watched.get(topic_lec[(subj, topic)])
                prep = 0.45 if w and w[0] <= day_s and w[1] >= 80 else 0.1 if w and w[0] <= day_s else -0.35
                p = _sigmoid(1.1 * skill + s["affinity"][subj] + prep - b + 0.3)
                if rng.random() < 0.07:
                    rows.append((s["id"], qid, None, None))
                    skipped += 1
                elif rng.random() < p:
                    rows.append((s["id"], qid, ans, 1))
                    correct += 1
                else:
                    wrong_opts = [o for o in range(4) if o != ans]
                    pick = popular if popular != ans and rng.random() < 0.55 else rng.choice(wrong_opts)
                    rows.append((s["id"], qid, pick, 0))
                    wrong += 1
            taken = max(8, min(mins, round(rng.gauss(0.82, 0.1) * mins)))
            tabs = rng.choices([0, 1, 2, 3], [70, 18, 8, 4])[0]
            t["answers"] += rows
            t["attempts"].append([s["id"], tid, taken, tabs, correct, wrong, skipped, 4 * correct - wrong])

    _plant_integrity(rng, t)
    stories = [{"student_id": sid, "story": story, "truth": truth} for sid, (story, truth) in STORIES.items()]
    t["attempts"] = [tuple(a) for a in t["attempts"]]
    return t, stories


def _plant_integrity(rng, t) -> None:
    """The copier pair in Grand Test 3, the open speed flag in Grand Test 5, the cleared tab-switch flag in Grand Test 2."""
    gt = {name: tid for tid, name, *_ in t["tests"]}
    answers = {(a[0], a[1]): a for a in t["answers"]}
    qs = [q for q in t["questions"] if q[1] == gt["Grand Test 3"]]
    for q in qs:  # S105 copies S131's choice on 95 of 100 questions
        src = answers[("S131", q[0])]
        if rng.random() < 0.95:
            answers[("S105", q[0])] = ("S105", q[0], src[2], src[3])
    t["answers"] = list(answers.values())
    att = {(a[0], a[1]): a for a in t["attempts"]}

    def rescore(sid, tid):
        rows = [answers[(sid, q[0])] for q in t["questions"] if q[1] == tid]
        c = sum(r[3] == 1 for r in rows)
        w = sum(r[3] == 0 for r in rows)
        att[(sid, tid)][4:8] = [c, w, len(rows) - c - w, 4 * c - w]
    rescore("S105", gt["Grand Test 3"])
    att[("S105", gt["Grand Test 3"])][3] = 6

    for q in [q for q in t["questions"] if q[1] == gt["Grand Test 5"]]:  # S150: near-perfect in 21 minutes
        answers[("S150", q[0])] = ("S150", q[0], q[7], 1) if rng.random() < 0.94 else ("S150", q[0], (q[7] + 1) % 4, 0)
    t["answers"] = list(answers.values())
    rescore("S150", gt["Grand Test 5"])
    att[("S150", gt["Grand Test 5"])][2:4] = [21, 17]
    att[("S177", gt["Grand Test 2"])][3] = 12

    name = {s[0]: s[1] for s in t["students"]}
    t["integrity_reviews"] = [
        ("S105", gt["Grand Test 3"], "2026-08-10", "Same answers as another student", "confirmed", "2026-08-20",
         f"Shared almost every answer, wrong ones included, with {name['S131']}. Admitted copying; warned, score voided."),
        ("S131", gt["Grand Test 3"], "2026-08-10", "Same answers as another student", "cleared", "2026-08-20",
         f"Their answers were copied by {name['S105']}, a roommate; {name['S131']} was not involved."),
        ("S177", gt["Grand Test 2"], "2026-06-15", "12 tab switches", "cleared", "2026-06-22",
         "Power cut during the test; rejoined from phone. No sign of cheating."),
        ("S150", gt["Grand Test 5"], "2026-11-09", "Finished in 21 of 120 minutes with a near-top score; 17 tab switches",
         "open", None, "Awaiting the student's explanation."),
    ]


SCHEMA = """
CREATE TABLE batches (name TEXT PRIMARY KEY, mentor TEXT, starts_on TEXT);
CREATE TABLE students (id TEXT PRIMARY KEY, name TEXT, batch TEXT REFERENCES batches(name), joined_on TEXT);
CREATE TABLE lectures (id TEXT PRIMARY KEY, subject TEXT, title TEXT, course_order INTEGER, minutes INTEGER);
CREATE TABLE watch (student_id TEXT, lecture_id TEXT, first_watched TEXT, last_watched TEXT, watched_pct INTEGER,
                    PRIMARY KEY (student_id, lecture_id));
CREATE TABLE activity (student_id TEXT, day TEXT, minutes INTEGER, PRIMARY KEY (student_id, day));
CREATE TABLE tests (id TEXT PRIMARY KEY, name TEXT, kind TEXT, subject TEXT, held_on TEXT, questions INTEGER, minutes INTEGER);
CREATE TABLE questions (id TEXT PRIMARY KEY, test_id TEXT, number INTEGER, subject TEXT, topic TEXT, kind TEXT,
                        difficulty TEXT, answer INTEGER);
CREATE TABLE attempts (student_id TEXT, test_id TEXT, minutes_taken INTEGER, tab_switches INTEGER, correct INTEGER,
                       wrong INTEGER, skipped INTEGER, score INTEGER, PRIMARY KEY (student_id, test_id));
CREATE TABLE answers (student_id TEXT, question_id TEXT, chosen INTEGER, correct INTEGER, PRIMARY KEY (student_id, question_id));
CREATE TABLE integrity_reviews (student_id TEXT, test_id TEXT, opened_on TEXT, signal TEXT, outcome TEXT, decided_on TEXT,
                                note TEXT);
CREATE INDEX answers_q ON answers (question_id);
"""


def write(tables: dict[str, list[tuple]], path: Path = DB) -> None:
    path.parent.mkdir(exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for name, rows in tables.items():
        if rows:
            con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(rows[0]))})", rows)
    con.commit()
    con.execute("VACUUM")
    con.close()


if __name__ == "__main__":
    tables, stories = build()
    write(tables)
    (ROOT / "data" / "stories.json").write_text(json.dumps(stories, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in tables.items()}, f"{DB.stat().st_size / 1e6:.1f} MB")
