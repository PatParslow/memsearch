"""Cross-idea similarity/synergy scan over all currently 'live' synthesis
ideas -- accepted, needs_review, AND rejected. A rejected idea isn't
necessarily dead forever: if it turns out to share ground with, or is a
prerequisite for, a live idea, it may be worth resurrecting alongside it
rather than staying discarded.

Embeds each idea's own concept text (or, for a rejected idea, the
model's own reasoning for rejecting it -- see synthesis.py's
_write_report, which now persists that reasoning instead of discarding
it) using the same embedding model memsearch already uses everywhere
else, computes full pairwise cosine similarity (cheap at this scale --
a couple dozen ideas at most, nothing like the whole corpus), and for
the most similar pairs asks the local model to characterize what the
relationship actually is: shared technique, one foundational for
another, real synergy from combining them, or just superficial keyword
overlap with nothing underneath.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import graph as graph_mod
from . import refine, store, synthesis

SYNERGY_PATH = Path(graph_mod.DEFAULT_GRAPH_PATH).parent / "synthesis_synergies.json"
PROJECTS_PATH = Path(graph_mod.DEFAULT_GRAPH_PATH).parent / "synthesis_projects.json"
INDEX_PATH = Path(graph_mod.DEFAULT_GRAPH_PATH).parent / "synthesis_index.json"
# Calibrated against real data, not guessed: idea-concept similarity
# runs much lower than raw-document similarity in the main corpus (a
# handful of short, heterogeneous abstractive summaries across very
# different fields, not passages that can share a lot of literal
# vocabulary) -- the highest pair among the first 13 real ideas was
# 0.455, and 0.5 as an initial guess would have surfaced zero pairs.
MIN_SIMILARITY_DEFAULT = 0.3
TOP_N_DEFAULT = 15

RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationship": {
            "type": "string",
            "enum": ["shares_technique", "foundational", "synergistic", "superficial_only", "duplicate"],
            "description": (
                "shares_technique: both rely on the same underlying mechanism/method, applied to "
                "different problems. foundational: one idea's mechanism is a genuine prerequisite "
                "the other could build on. synergistic: combining them would likely produce "
                "something neither achieves alone. superficial_only: similar wording/domain but no "
                "real technical connection. duplicate: substantially the same idea twice."
            ),
        },
        "direction": {
            "type": "string",
            "enum": ["a_before_b", "b_before_a", "neither"],
            "description": "For 'foundational' only: which idea would need to come first. 'neither' if not applicable.",
        },
        "explanation": {"type": "string", "description": "2-4 sentences on the actual relationship"},
        "worth_resurrecting": {
            "type": "boolean",
            "description": "True only if one of the pair was rejected and this relationship gives it new grounds to reconsider",
        },
    },
    "required": ["relationship", "direction", "explanation", "worth_resurrecting"],
}


def _get_rejected_reasoning(report_text: str, title: str) -> str:
    """Pulls a rejected idea's own reasoning out of the "Considered but
    rejected" section, matched by title -- see synthesis.py's
    _write_report, which persists this text rather than only a bullet."""
    section_m = re.search(r"## Considered but rejected.*?\n\n(.*?)\Z", report_text, re.DOTALL)
    if not section_m:
        return ""
    section = section_m.group(1)
    title_a = title.split(" <-> ")[0].split(" (")[0].strip()
    for block in re.split(r"(?m)^- \*\*", section)[1:]:
        heading, _, rest = block.partition("\n")
        if title_a in heading:
            return rest.strip()
    return ""


def _gather_ideas() -> list[dict]:
    index = json.loads(SYNERGY_PATH.parent.joinpath("synthesis_index.json").read_text(encoding="utf-8")) \
        if SYNERGY_PATH.parent.joinpath("synthesis_index.json").exists() else []
    report_cache: dict[str, str] = {}
    ideas = []
    for entry in index:
        report_path = entry.get("report_path")
        if not report_path or not Path(report_path).is_file():
            continue
        if report_path not in report_cache:
            report_cache[report_path] = Path(report_path).read_text(encoding="utf-8")
        text = report_cache[report_path]

        if entry.get("accepted"):
            state = refine._extract_idea_block(text, entry["idea_number"])
            content = state["concept"] if state else ""
        else:
            content = _get_rejected_reasoning(text, entry.get("title", ""))

        if not content.strip():
            continue
        ideas.append({
            "title": entry.get("title", "?"),
            "report_file": entry.get("report_file"),
            "report_path": report_path,
            "idea_number": entry.get("idea_number"),
            "status": entry.get("status", "accepted" if entry.get("accepted") else "rejected"),
            "content": content,
        })
    return ideas


def _build_relationship_prompt(idea_a: dict, idea_b: dict) -> str:
    return f"""Two ideas from a personal knowledge base's idea-synthesis pipeline were found to be similar in embedding space. Determine what the actual relationship is.

IDEA A ({idea_a['status']}): {idea_a['title']}
{idea_a['content']}

IDEA B ({idea_b['status']}): {idea_b['title']}
{idea_b['content']}

Characterize the relationship honestly -- most pairs that merely share domain vocabulary have no real technical connection underneath ("superficial_only" is a legitimate and common answer, not a failure to find something). Only call it "foundational" if one idea's actual mechanism is something the other could concretely build on, not just related-in-theme. If one of the two is "rejected", only mark worth_resurrecting=true if this specific relationship gives it real new grounds to reconsider -- not merely because it's now sitting next to a live idea."""


def _characterize(idea_a: dict, idea_b: dict) -> dict | None:
    prompt = _build_relationship_prompt(idea_a, idea_b)

    def is_valid(r: dict) -> bool:
        return len(str(r.get("explanation", "")).split()) >= 8

    return synthesis._ollama_generate_structured(prompt, RELATIONSHIP_SCHEMA, is_valid)


def run_synergy_scan(min_similarity: float = MIN_SIMILARITY_DEFAULT, top_n: int = TOP_N_DEFAULT) -> Path:
    ideas = _gather_ideas()
    if len(ideas) < 2:
        raise RuntimeError("Need at least 2 ideas with real content to compare")

    print(f"Embedding {len(ideas)} ideas...", flush=True)
    vectors = np.array(store._EMBEDDING_FUNCTION([i["content"] for i in ideas]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1e-9
    unit = vectors / norms
    sim = unit @ unit.T

    pairs = []
    for i in range(len(ideas)):
        for j in range(i + 1, len(ideas)):
            s = float(sim[i, j])
            if s >= min_similarity:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    pairs = pairs[:top_n]
    print(f"  -> {len(pairs)} pair(s) above similarity {min_similarity:.2f} (of "
          f"{len(ideas) * (len(ideas) - 1) // 2} possible)", flush=True)

    results = []
    for k, (s, i, j) in enumerate(pairs, 1):
        a, b = ideas[i], ideas[j]
        print(f"  [{k}/{len(pairs)}] {a['title'][:40]} <-> {b['title'][:40]} (sim {s:.2f})...", flush=True)
        rel = _characterize(a, b)
        if not rel:
            print("      ! characterization failed, skipping", flush=True)
            continue
        results.append({
            "idea_a": {"report_file": a["report_file"], "idea_number": a["idea_number"], "title": a["title"], "status": a["status"]},
            "idea_b": {"report_file": b["report_file"], "idea_number": b["idea_number"], "title": b["title"], "status": b["status"]},
            "similarity": round(s, 3),
            **rel,
        })

    SYNERGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNERGY_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return SYNERGY_PATH


CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "parent_title": {"type": "string", "description": "A short (3-8 word) name for the project uniting these ideas"},
        "parent_description": {"type": "string", "description": "3-6 sentences describing the unified project and how the child ideas fit together as parts of it"},
        "suggested_sequence": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "reason": {"type": "string"},
            }, "required": ["title", "reason"]},
            "description": "The child ideas in a sensible build order, each with a short reason for that position",
        },
    },
    "required": ["parent_title", "parent_description", "suggested_sequence"],
}


def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_index(index: list[dict]) -> None:
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def consolidate_ideas(children: list[tuple[str, int]]) -> dict:
    """children: list of (report_file, idea_number) pairs, typically ones
    the synergy scan already flagged as foundational/synergistic with
    each other. Gathers each child's current concept, asks the local
    model to synthesize a parent project that unifies them (title,
    description, a sensible build order), records it in
    synthesis_projects.json, and sets parent_id on each child's index
    entry so the board can group them."""
    index = _load_index()
    by_key = {(e.get("report_file"), e.get("idea_number")): e for e in index}

    child_details = []
    report_cache: dict[str, str] = {}
    for report_file, idea_number in children:
        entry = by_key.get((report_file, idea_number))
        if not entry:
            raise RuntimeError(f"No index entry for {report_file} idea {idea_number}")
        if entry.get("report_path") not in report_cache:
            report_cache[entry["report_path"]] = Path(entry["report_path"]).read_text(encoding="utf-8")
        state = refine._extract_idea_block(report_cache[entry["report_path"]], idea_number)
        child_details.append({
            "report_file": report_file, "idea_number": idea_number,
            "title": entry.get("title", "?"), "concept": state["concept"] if state else "",
        })

    listing = "\n\n".join(
        f"CHILD {i}: {c['title']}\n{c['concept']}" for i, c in enumerate(child_details, 1)
    )
    prompt = f"""These ideas were flagged as related (shared technique, foundational dependency, or synergy) by an automated scan of a personal knowledge base's idea-synthesis pipeline. Synthesize a single parent project that unifies them.

{listing}

Give the parent project a name and description that genuinely reflects what unites these specific ideas (not a generic umbrella term), and propose a sensible order to actually build them in, explaining why each one comes where it does (e.g. what it depends on from an earlier one)."""

    def is_valid(r: dict) -> bool:
        return (
            len(str(r.get("parent_title", "")).split()) >= 2
            and len(str(r.get("parent_description", "")).split()) >= 15
            and isinstance(r.get("suggested_sequence"), list) and len(r["suggested_sequence"]) > 0
        )

    result = synthesis._ollama_generate_structured(prompt, CONSOLIDATE_SCHEMA, is_valid)
    if not result:
        raise RuntimeError("Local model failed to produce a valid parent-project synthesis")

    projects = []
    if PROJECTS_PATH.exists():
        try:
            projects = json.loads(PROJECTS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            projects = []
    project_id = f"proj_{len(projects) + 1}"
    project = {
        "id": project_id,
        "title": result["parent_title"],
        "description": result["parent_description"],
        "suggested_sequence": result["suggested_sequence"],
        "children": [{"report_file": c["report_file"], "idea_number": c["idea_number"], "title": c["title"]} for c in child_details],
    }
    projects.append(project)
    PROJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_PATH.write_text(json.dumps(projects, indent=2), encoding="utf-8")

    for report_file, idea_number in children:
        entry = by_key.get((report_file, idea_number))
        if entry:
            entry["parent_id"] = project_id
    _save_index(index)

    return project
