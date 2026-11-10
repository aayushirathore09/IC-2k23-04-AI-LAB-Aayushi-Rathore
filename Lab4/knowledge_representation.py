# %% [markdown]
# # Lab 4: Knowledge Representation & Logical Reasoning
# **Course:** IC-709D | AI Lab  **Unit 4 (CO1, CO3)**
#
# **Activities:** University Knowledge Base, Medical Knowledge Representation,
# Semantic Network Construction, Frame Representation, Logical Rule Engine.
# **Applications:** Expert System Knowledge Base, Intelligent Recommendation Engine.
# **Analysis:** Frames vs Semantic Networks vs Rule-Based Systems.

# %% [code] imports
"""Lab 4 - Knowledge Representation & Logical Reasoning."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

Fact = Tuple[str, ...]          # e.g. ("enrolled", "Aman", "AI")
Bindings = Dict[str, str]       # e.g. {"?s": "Aman"}

# %% [markdown]
# ## Part 1: Logical Rule Engine (core of the lab)
# Facts are predicates stored as tuples. Variables start with `?`.
# The engine supports **forward chaining** (data-driven) and
# **backward chaining** (goal-driven) using unification.

# %% [code] rule engine
def is_var(term: str) -> bool:
    """Return True if the term is a logic variable (starts with '?')."""
    return isinstance(term, str) and term.startswith("?")


def walk(term: str, bindings: Bindings) -> str:
    """Follow variable bindings until a constant or unbound variable is found."""
    while is_var(term) and term in bindings:
        term = bindings[term]
    return term


def unify(p: Fact, q: Fact, bindings: Bindings) -> Optional[Bindings]:
    """Unify two predicates; return extended bindings or None on failure."""
    if len(p) != len(q):
        return None
    result = dict(bindings)
    for a, b in zip(p, q):
        a, b = walk(a, result), walk(b, result)
        if a == b:
            continue
        if is_var(a):
            result[a] = b
        elif is_var(b):
            result[b] = a
        else:
            return None
    return result


def substitute(pattern: Fact, bindings: Bindings) -> Fact:
    """Replace bound variables in a pattern with their values."""
    return tuple(walk(t, bindings) for t in pattern)


@dataclass
class Rule:
    """IF all `conditions` hold THEN `conclusion` holds."""
    name: str
    conditions: List[Fact]
    conclusion: Fact

    def renamed(self, suffix: str) -> "Rule":
        """Standardise variables apart (needed for backward chaining)."""
        ren = lambda f: tuple(t + suffix if is_var(t) else t for t in f)
        return Rule(self.name, [ren(c) for c in self.conditions], ren(self.conclusion))


def satisfy(conditions: List[Fact], facts: set, bindings: Bindings) -> Iterator[Bindings]:
    """Yield every binding that makes all conditions true against `facts`."""
    if not conditions:
        yield bindings
        return
    first, rest = conditions[0], conditions[1:]
    for fact in list(facts):
        new = unify(first, fact, bindings)
        if new is not None:
            yield from satisfy(rest, facts, new)


def forward_chain(facts: set, rules: List[Rule]) -> Tuple[set, List[str]]:
    """Apply rules repeatedly until no new fact is derived.

    Returns the enlarged fact set and a trace explaining each derivation.
    """
    known, trace = set(facts), []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            for b in satisfy(rule.conditions, known, {}):
                new_fact = substitute(rule.conclusion, b)
                if any(is_var(t) for t in new_fact):
                    continue  # unsafe rule, skip ungrounded conclusions
                if new_fact not in known:
                    known.add(new_fact)
                    trace.append(f"[{rule.name}] => {new_fact}")
                    changed = True
    return known, trace


def backward_chain(goals: List[Fact], facts: set, rules: List[Rule],
                   bindings: Optional[Bindings] = None, depth: int = 0,
                   max_depth: int = 10) -> Iterator[Bindings]:
    """Goal-driven proof (SLD resolution). Yields bindings proving all goals."""
    bindings = bindings or {}
    if not goals:
        yield bindings
        return
    if depth > max_depth:
        return
    goal, rest = substitute(goals[0], bindings), goals[1:]
    for fact in list(facts):                       # 1) match known facts
        b = unify(goal, fact, bindings)
        if b is not None:
            yield from backward_chain(rest, facts, rules, b, depth, max_depth)
    for rule in rules:                             # 2) expand via rules
        r = rule.renamed(f"_{depth}")
        b = unify(goal, r.conclusion, bindings)
        if b is not None:
            yield from backward_chain(r.conditions + rest, facts, rules, b,
                                      depth + 1, max_depth)


def ask(goal: Fact, facts: set, rules: List[Rule]) -> List[Fact]:
    """Return all distinct ground answers to `goal` using backward chaining."""
    answers = []
    for b in backward_chain([goal], facts, rules):
        ans = substitute(goal, b)
        if ans not in answers:
            answers.append(ans)
    return answers

# %% [markdown]
# ## Activity 1: University Knowledge Base
# Predicate representation of students, courses, teachers and prerequisites,
# plus rules to infer eligibility, classmates and instructors.

# %% [code] university kb
def build_university_kb() -> Tuple[set, List[Rule]]:
    """Create facts and inference rules for the university domain."""
    facts = {
        ("student", "Aman"), ("student", "Riya"), ("student", "Karan"),
        ("course", "Python"), ("course", "AI"), ("course", "ML"),
        ("no_prereq", "Python"),
        ("prerequisite", "AI", "Python"), ("prerequisite", "ML", "AI"),
        ("completed", "Aman", "Python"), ("completed", "Aman", "AI"),
        ("completed", "Riya", "Python"),
        ("enrolled", "Riya", "AI"), ("enrolled", "Karan", "AI"),
        ("enrolled", "Aman", "ML"),
        ("teaches", "Dr_Sharma", "AI"), ("teaches", "Dr_Verma", "ML"),
        ("teaches", "Dr_Rao", "Python"),
    }
    rules = [
        Rule("R1 eligible-no-prereq", [("student", "?s"), ("no_prereq", "?c")],
             ("eligible", "?s", "?c")),
        Rule("R2 eligible-with-prereq",
             [("student", "?s"), ("prerequisite", "?c", "?p"), ("completed", "?s", "?p")],
             ("eligible", "?s", "?c")),
        Rule("R3 classmates", [("enrolled", "?a", "?c"), ("enrolled", "?b", "?c")],
             ("classmate", "?a", "?b")),
        Rule("R4 instructor", [("enrolled", "?s", "?c"), ("teaches", "?t", "?c")],
             ("taught_by", "?s", "?t")),
    ]
    return facts, rules


def demo_university() -> None:
    facts, rules = build_university_kb()
    print("Q1. Which courses is Riya eligible for?")
    for ans in ask(("eligible", "Riya", "?c"), facts, rules):
        print("   ", ans)
    print("Q2. Who is taught by Dr_Sharma?")
    for ans in ask(("taught_by", "?s", "Dr_Sharma"), facts, rules):
        print("   ", ans)
    print("Q3. Classmates of Riya (excluding herself):")
    for ans in ask(("classmate", "Riya", "?x"), facts, rules):
        if ans[2] != "Riya":
            print("   ", ans)

# %% [markdown]
# ## Activity 2: Medical Knowledge Representation
# Diseases are represented as a knowledge base of symptoms. A weighted
# matching rule ranks the most likely conditions.
# *(Educational only - not a real medical diagnosis tool.)*

# %% [code] medical kb
MEDICAL_KB: Dict[str, Dict[str, Any]] = {
    "Common Cold": {"symptoms": {"cough", "sneezing", "runny_nose", "sore_throat"},
                    "advice": "Rest, fluids, warm drinks."},
    "Influenza": {"symptoms": {"fever", "cough", "body_ache", "fatigue", "headache"},
                  "advice": "Rest, fluids, consult a doctor if severe."},
    "Malaria": {"symptoms": {"fever", "chills", "sweating", "headache", "nausea"},
                "advice": "Get a blood test urgently."},
    "Migraine": {"symptoms": {"headache", "nausea", "light_sensitivity"},
                 "advice": "Rest in a dark room, consult a neurologist."},
    "COVID-19": {"symptoms": {"fever", "cough", "loss_of_taste", "fatigue", "breathlessness"},
                 "advice": "Isolate and get tested."},
}


def diagnose(symptoms: set, kb: Dict[str, Dict[str, Any]] = MEDICAL_KB
             ) -> List[Tuple[str, float]]:
    """Rank diseases by fraction of their symptoms present (descending)."""
    if not symptoms:
        raise ValueError("Please provide at least one symptom.")
    symptoms = {s.strip().lower().replace(" ", "_") for s in symptoms}
    unknown = symptoms - set().union(*(d["symptoms"] for d in kb.values()))
    if unknown:
        print(f"   (warning: unknown symptoms ignored: {sorted(unknown)})")
    scores = [(name, len(symptoms & d["symptoms"]) / len(d["symptoms"]))
              for name, d in kb.items()]
    return sorted([s for s in scores if s[1] > 0], key=lambda x: -x[1])


def demo_medical() -> None:
    patient = {"fever", "cough", "fatigue", "headache"}
    print("Patient symptoms:", sorted(patient))
    results = diagnose(patient)
    for disease, score in results:
        print(f"   {disease:<12} match = {score:.0%}")
    print("Advice for top match:", MEDICAL_KB[results[0][0]]["advice"])
    plt.figure(figsize=(6, 3))
    plt.barh([r[0] for r in results][::-1], [r[1] for r in results][::-1], color="teal")
    plt.xlabel("Symptom match")
    plt.title("Medical knowledge base: diagnosis ranking")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diagnosis_scores.png", dpi=120)
    plt.show()

# %% [markdown]
# ## Activity 3: Semantic Network Construction
# Nodes are concepts, labelled edges are relations (`is_a`, `has`, `can`, ...).
# Properties are **inherited** along `is_a` links, with exceptions overriding.

# %% [code] semantic network
class SemanticNetwork:
    """Directed labelled graph with is-a inheritance."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add(self, source: str, relation: str, target: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def parents(self, node: str) -> List[str]:
        return [t for _, t, d in self.graph.out_edges(node, data=True)
                if d["relation"] == "is_a"]

    def properties(self, node: str) -> Dict[str, List[str]]:
        """Collect relation -> values, inherited via is_a.

        A `cannot X` link on a child cancels an inherited `can X`.
        """
        if node not in self.graph:
            raise KeyError(f"Unknown concept: {node}")
        props: Dict[str, List[str]] = {}
        for parent in self.parents(node):
            for rel, vals in self.properties(parent).items():
                props.setdefault(rel, []).extend(v for v in vals if v not in props.get(rel, []))
        for _, t, d in self.graph.out_edges(node, data=True):
            rel = d["relation"]
            if rel == "is_a":
                continue
            if rel == "cannot":
                props["can"] = [v for v in props.get("can", []) if v != t]
            elif t not in props.setdefault(rel, []):
                props[rel].append(t)
        return {k: v for k, v in props.items() if v}

    def draw(self, filename: str = "semantic_network.png") -> None:
        pos = nx.spring_layout(self.graph, seed=7, k=1.2)
        plt.figure(figsize=(10, 7))
        nx.draw(self.graph, pos, with_labels=True, node_color="#bde0fe",
                node_size=2400, font_size=9, arrowsize=18)
        labels = {(u, v): d["relation"] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels,
                                     font_size=8, font_color="firebrick")
        plt.title("Semantic Network: Animals")
        plt.savefig(OUTPUT_DIR / filename, dpi=120, bbox_inches="tight")
        plt.show()


def build_animal_network() -> SemanticNetwork:
    net = SemanticNetwork()
    for s, r, t in [
        ("Animal", "has", "Cells"), ("Animal", "can", "Move"),
        ("Bird", "is_a", "Animal"), ("Bird", "has", "Wings"), ("Bird", "can", "Fly"),
        ("Penguin", "is_a", "Bird"), ("Penguin", "can", "Swim"),
        ("Fish", "is_a", "Animal"), ("Fish", "has", "Gills"),
        ("Mammal", "is_a", "Animal"), ("Dog", "is_a", "Mammal"),
        ("Tweety", "is_a", "Bird"), ("Tux", "is_a", "Penguin"),
    ]:
        net.add(s, r, t)
    net.add("Penguin", "cannot", "Fly")   # exception overrides inherited 'can Fly'
    return net


def demo_semantic_network() -> None:
    net = build_animal_network()
    for concept in ("Tweety", "Tux", "Dog"):
        print(f"{concept}: {net.properties(concept)}")
    net.draw()

# %% [markdown]
# ## Activity 4: Frame Representation
# A **frame** is a structured record with *slots*. Slot values come from
# (1) the frame itself, (2) an *if-needed* procedure, (3) a *default*,
# or (4) the parent frame (inheritance).

# %% [code] frames
@dataclass
class Frame:
    name: str
    parent: Optional["Frame"] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    if_needed: Dict[str, Callable[["Frame"], Any]] = field(default_factory=dict)

    def get(self, slot: str, origin: Optional["Frame"] = None) -> Any:
        """Look up a slot; `origin` is the frame the query started from."""
        origin = origin or self
        if slot in self.slots:
            return self.slots[slot]
        if slot in self.if_needed:
            return self.if_needed[slot](origin)   # run procedure on the instance
        if slot in self.defaults:
            return self.defaults[slot]
        if self.parent:
            return self.parent.get(slot, origin)
        raise KeyError(f"Slot '{slot}' not found in frame '{origin.name}'")

    def show(self) -> None:
        print(f"Frame<{self.name}>" + (f"  (is-a {self.parent.name})" if self.parent else ""))
        for slot in ("name", "age", "course", "marks", "grade", "has_id_card"):
            try:
                print(f"    {slot:<12}: {self.get(slot)}")
            except KeyError:
                pass


def grade_from_marks(frame: Frame) -> str:
    """If-needed procedure: compute grade from marks."""
    marks = frame.get("marks")
    return "A" if marks >= 80 else "B" if marks >= 60 else "C" if marks >= 40 else "F"


def demo_frames() -> None:
    person = Frame("Person", defaults={"has_id_card": False})
    student = Frame("Student", parent=person,
                    defaults={"has_id_card": True},
                    if_needed={"grade": grade_from_marks})
    mca = Frame("MCA_Student", parent=student, slots={"course": "MCA"})
    rahul = Frame("Rahul", parent=mca, slots={"name": "Rahul", "age": 23, "marks": 86})
    meena = Frame("Meena", parent=mca, slots={"name": "Meena", "age": 22, "marks": 55})
    for f in (rahul, meena):
        f.show()
    try:
        rahul.get("salary")
    except KeyError as err:
        print("Handled missing slot ->", err)

# %% [markdown]
# ## Application 1: Expert System Knowledge Base
# A forward-chaining expert system for **computer fault diagnosis**, with an
# explanation facility (it shows *which rules fired*).

# %% [code] expert system
def build_expert_kb() -> List[Rule]:
    return [
        Rule("E1", [("symptom", "no_power_light")], ("cause", "power_supply_issue")),
        Rule("E2", [("symptom", "beeps_on_boot"), ("symptom", "no_display")],
             ("cause", "ram_fault")),
        Rule("E3", [("symptom", "slow_performance"), ("symptom", "high_disk_usage")],
             ("cause", "failing_hdd_or_malware")),
        Rule("E4", [("symptom", "overheating"), ("symptom", "fan_noise")],
             ("cause", "clogged_fan")),
        Rule("E5", [("cause", "power_supply_issue")], ("fix", "check_charger_and_battery")),
        Rule("E6", [("cause", "ram_fault")], ("fix", "reseat_or_replace_ram")),
        Rule("E7", [("cause", "failing_hdd_or_malware")], ("fix", "run_antivirus_and_backup_data")),
        Rule("E8", [("cause", "clogged_fan")], ("fix", "clean_fan_and_use_cooling_pad")),
    ]


def demo_expert_system() -> None:
    observed = {("symptom", "overheating"), ("symptom", "fan_noise"),
                ("symptom", "slow_performance"), ("symptom", "high_disk_usage")}
    known, trace = forward_chain(observed, build_expert_kb())
    print("Explanation (rules fired):")
    for line in trace:
        print("   ", line)
    print("Recommended fixes:", [f[1] for f in known if f[0] == "fix"])

# %% [markdown]
# ## Application 2: Intelligent Recommendation Engine
# Combines the **University KB** (eligibility rules) with an interest-matching
# score to recommend courses a student can actually take.

# %% [code] recommendation engine
COURSE_TAGS = {
    "Python": {"programming", "basics"},
    "AI": {"ai", "search", "logic"},
    "ML": {"ai", "data", "prediction"},
}


def recommend_courses(student: str, interests: set, top_k: int = 3
                      ) -> List[Tuple[str, float]]:
    """Recommend eligible, not-yet-completed courses ranked by interest overlap."""
    facts, rules = build_university_kb()
    if ("student", student) not in facts:
        raise ValueError(f"Unknown student: {student}")
    interests = {i.lower() for i in interests}
    eligible = {a[2] for a in ask(("eligible", student, "?c"), facts, rules)}
    done = {a[2] for a in ask(("completed", student, "?c"), facts, rules)}
    scored = [(c, len(interests & COURSE_TAGS[c]) / len(COURSE_TAGS[c]))
              for c in eligible - done]
    return sorted(scored, key=lambda x: -x[1])[:top_k]


def demo_recommendation() -> None:
    for student, interests in [("Riya", {"ai", "logic"}), ("Aman", {"data", "prediction"})]:
        print(f"{student} (interests={sorted(interests)}):")
        recs = recommend_courses(student, interests)
        for course, score in recs or [("(nothing new to recommend)", 0)]:
            print(f"    {course:<28} score = {score:.2f}")

# %% [markdown]
# ## Performance & Analysis
# 1. Forward vs backward chaining execution time.
# 2. Comparison of Frames, Semantic Networks and Rule-Based Systems.

# %% [code] analysis
def benchmark_chaining(repeats: int = 200) -> pd.DataFrame:
    facts, rules = build_university_kb()
    t0 = time.perf_counter()
    for _ in range(repeats):
        forward_chain(facts, rules)
    fwd = (time.perf_counter() - t0) / repeats * 1000
    t0 = time.perf_counter()
    for _ in range(repeats):
        ask(("eligible", "Riya", "?c"), facts, rules)
    bwd = (time.perf_counter() - t0) / repeats * 1000
    return pd.DataFrame({"Method": ["Forward chaining (all inferences)",
                                    "Backward chaining (one goal)"],
                         "Avg time (ms)": [round(fwd, 3), round(bwd, 3)]})


def comparison_table() -> pd.DataFrame:
    return pd.DataFrame({
        "Aspect": ["Structure", "Inheritance", "Reasoning", "Best for",
                   "Weakness"],
        "Frames": ["Slots + values", "Yes, strong (with defaults)",
                   "Default/if-needed procedures", "Structured objects",
                   "Hard to express complex logic"],
        "Semantic Networks": ["Graph of nodes + labelled links", "Yes, via is_a links",
                              "Graph traversal", "Concept relationships",
                              "No standard semantics; exceptions are messy"],
        "Rule-Based Systems": ["IF-THEN rules", "No (not built in)",
                               "Forward/backward chaining", "Decision making, diagnosis",
                               "Many rules are hard to maintain"],
    })


def demo_analysis() -> None:
    bench = benchmark_chaining()
    print(bench.to_string(index=False))
    table = comparison_table()
    print(table.to_string(index=False))
    bench.to_csv(OUTPUT_DIR / "chaining_benchmark.csv", index=False)
    table.to_csv(OUTPUT_DIR / "kr_comparison.csv", index=False)

# %% [code] main
def main() -> None:
    sections = [("1. University Knowledge Base", demo_university),
                ("2. Medical Knowledge Representation", demo_medical),
                ("3. Semantic Network", demo_semantic_network),
                ("4. Frame Representation", demo_frames),
                ("5. Expert System (Application)", demo_expert_system),
                ("6. Recommendation Engine (Application)", demo_recommendation),
                ("7. Analysis", demo_analysis)]
    for title, func in sections:
        print(f"\n===== {title} =====")
        func()


if __name__ == "__main__":
    main()
