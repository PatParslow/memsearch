"""LLM-driven cross-domain idea synthesis over the knowledge graph's own
interpolation gaps -- pairs of nodes that are related in embedding space
but have nothing bridging them. Turns each gap from a numeric flag into
a written, reviewable proposal: is there a real connection here, and if
so, what concrete thing would test it.

Two sequential Ollama calls per gap (concept, then test design), both
with the model's own reasoning kept and written into the report -- a
confident final answer built on reasoning that doesn't actually engage
with the source material is a real tell that the idea shouldn't be
trusted, so the reasoning is treated as real output, not overhead. This
is why a large "thinking" model is used here despite the latency cost
(order of a minute per call) -- unlike the title generator in graph.py,
which explicitly wants a fast, terse answer, this task explicitly wants
the trace visible for human review.

Every report is unambiguously labelled as machine-generated speculation,
both in the file's own header and via the project name ("synthesis") it
gets mined under, so it can never be mistaken for verified fact or
human-authored work when it turns up in a future search.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import graph as graph_mod
from . import store

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
SYNTHESIS_MODEL = "qwen3.8-iq3xxs-64k:latest"
SYNTHESIS_TIMEOUT = 300  # seconds -- thinking calls on a 27B model run ~60-90s; leave real headroom

SYNTHESIS_DIR = Path(__file__).parent.parent / "synthesis"
SAMPLE_CHAR_BUDGET = 1500

CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Step-by-step reasoning about whether a genuine, substantive connection exists between the two areas"},
        "has_substantive_connection": {"type": "boolean"},
        "proposed_concept": {"type": "string", "description": "A 3-5 sentence concrete cross-over idea. Empty string if has_substantive_connection is false."},
    },
    "required": ["reasoning", "has_substantive_connection", "proposed_concept"],
}

TEST_DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Reasoning about what would actually validate or falsify the idea, and what a minimum viable test looks like"},
        "what_to_build": {"type": "string", "description": "A concrete, minimal thing to build -- small enough to actually run, not a research program"},
        "success_metrics": {"type": "string", "description": "Specific, measurable success criteria"},
        "how_to_falsify": {"type": "string", "description": "What result would show the idea doesn't hold up"},
        "open_questions": {
            "type": "array", "items": {"type": "string"},
            "description": "2-4 concrete, checkable claims a literature/web search could support or refute -- empty array if the idea rests entirely on the two areas' own already-mined content with nothing external to check",
        },
    },
    "required": ["reasoning", "what_to_build", "success_metrics", "how_to_falsify", "open_questions"],
}


def _ollama_generate_structured(prompt: str, schema: dict, is_valid, attempts: int = 3) -> dict | None:
    # `is_valid(result) -> bool` catches a failure mode seen in practice on
    # a long/complex prompt that plain exception-handling can't: the model
    # wrote its entire polished answer inside the free-form `reasoning`
    # field, then had nothing left to say once grammar-constrained decoding
    # forced it into the structured fields -- valid JSON, schema-conformant,
    # but blank. Garbage-in-garbage-out into stage 2 (or the report) is
    # worse than a retry.
    # Ollama's structured-outputs support (format=<json-schema>) lets the
    # model think freely first, then grammar-constrains only the final
    # answer to valid JSON matching the schema -- this replaced an earlier
    # heading-based markdown parser that broke on real output: the model
    # rehearsed the requested "## Heading" structure 2-3 times while
    # thinking before its real final answer, and a first-match regex
    # extracted the draft rehearsal instead of the final version.
    body = json.dumps({
        "model": SYNTHESIS_MODEL, "prompt": prompt, "format": schema, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_GENERATE_URL, data=body, headers={"Content-Type": "application/json"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=SYNTHESIS_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result = json.loads(data["response"])
        except Exception as e:
            print(f"    ! Ollama call failed (attempt {attempt + 1}/{attempts}): {e}", flush=True)
            continue
        if not is_valid(result):
            print(f"    ! Degenerate (blank field) response, retrying "
                  f"(attempt {attempt + 1}/{attempts})", flush=True)
            continue
        return result
    return None


def _sample_text_for_node(node_id: str, nodes_by_id: dict, col) -> str:
    node = nodes_by_id.get(node_id)
    if not node:
        return "(no sample text available)"
    res = col.get(where={"source_file": node["source_file"]}, include=["documents"], limit=5)
    docs = res.get("documents") or []
    joined = "\n\n".join(docs)
    return joined[:SAMPLE_CHAR_BUDGET] if joined else "(no sample text available)"


def _pick_top_gaps(graph_data: dict, limit: int, offset: int = 0) -> list[dict]:
    gaps = sorted(graph_data.get("interpolation_gaps", []), key=lambda g: g["gap_score"], reverse=True)
    return gaps[offset : offset + limit]


def _propose_concept(node_a: dict, node_b: dict, sample_a: str, sample_b: str) -> dict:
    prompt = f"""You are analyzing a personal knowledge base for genuine cross-pollination opportunities between two areas of work that are related in embedding space but have no direct bridge between them.

AREA A: "{node_a['title']}" (project: {node_a['project']})
Sample content:
{sample_a}

AREA B: "{node_b['title']}" (project: {node_b['project']})
Sample content:
{sample_b}

Think through whether there is a genuine, substantive connection here -- not just superficial keyword overlap. Be honest if the connection seems weak or contrived rather than forcing an idea to sound impressive; it is fine and useful to conclude there isn't a real connection."""

    def is_valid(r: dict) -> bool:
        if not str(r.get("reasoning", "")).strip():
            return False
        concept = r.get("proposed_concept", "").strip()
        # An empty proposed_concept is a legitimate answer when
        # has_substantive_connection is false -- only degenerate if it's
        # blank despite claiming a connection was found.
        if not r.get("has_substantive_connection"):
            return True
        if not concept:
            return False
        # Seen in practice: multiple draft title/framing alternatives
        # crammed into one field with " | " separators instead of the
        # requested single clean paragraph, on a call whose `reasoning`
        # was suspiciously short -- the model apparently skipped its
        # usual deliberation and dumped candidates into the answer field
        # instead of picking one. Not caught by an emptiness check.
        return concept.count(" | ") < 2

    result = _ollama_generate_structured(prompt, CONCEPT_SCHEMA, is_valid)
    if not result:
        return {"reasoning": None, "concept": None, "has_connection": False}
    has_connection = bool(result.get("has_substantive_connection")) and bool(result.get("proposed_concept", "").strip())
    return {
        "reasoning": result.get("reasoning"),
        "concept": result.get("proposed_concept") if has_connection else None,
        "has_connection": has_connection,
    }


def _propose_test_design(concept: str) -> dict:
    prompt = f"""A candidate cross-over idea has been proposed for a personal knowledge base:

{concept}

Design a concrete, minimal system to test this idea -- something small enough to actually build and run, not a research program.

You have no ability to search the web or check literature -- if the idea's plausibility depends on something you cannot verify from first principles alone (whether prior art already exists, whether a cited technique actually behaves the way assumed, whether a claimed effect is established or contested), name that explicitly as an open question rather than asserting it either way."""

    def is_valid(r: dict) -> bool:
        # A minimum word count, not just non-empty, matters here: a real
        # run produced literal placeholder category names as the field
        # values ("What to build: Minimal system design" / "Success
        # metrics: Concrete implementation") -- non-empty, schema-valid,
        # but content-free. Every genuine answer observed in practice runs
        # well over a dozen words; 8 is a conservative floor beneath that.
        fields = ("reasoning", "what_to_build", "success_metrics", "how_to_falsify")
        return all(len(str(r.get(f, "")).split()) >= 8 for f in fields)

    result = _ollama_generate_structured(prompt, TEST_DESIGN_SCHEMA, is_valid)
    if not result:
        return {"reasoning": None, "design": None, "open_questions": []}
    design = (
        f"- **What to build:** {result.get('what_to_build', '')}\n"
        f"- **Success metrics:** {result.get('success_metrics', '')}\n"
        f"- **How to falsify it:** {result.get('how_to_falsify', '')}"
    )
    return {
        "reasoning": result.get("reasoning"),
        "design": design,
        "open_questions": result.get("open_questions") or [],
    }


def run_synthesis(limit: int = 8, offset: int = 0, graph_path: str = graph_mod.DEFAULT_GRAPH_PATH) -> Path:
    graph_data = graph_mod.load_graph(graph_path)
    nodes_by_id = {n["id"]: n for n in graph_data["nodes"]}
    gaps = _pick_top_gaps(graph_data, limit, offset)
    if not gaps:
        raise RuntimeError("No interpolation gaps found in the graph -- run `memsearch graph build` first")

    col = store.get_collection()
    accepted: list[dict] = []
    rejected: list[dict] = []

    for i, gap in enumerate(gaps, 1):
        node_a, node_b = nodes_by_id.get(gap["node_a"]), nodes_by_id.get(gap["node_b"])
        if not node_a or not node_b:
            continue
        print(f"  [{i}/{len(gaps)}] {node_a['title']} <-> {node_b['title']} "
              f"(score {gap['gap_score']:.2f})...", flush=True)

        sample_a = _sample_text_for_node(gap["node_a"], nodes_by_id, col)
        sample_b = _sample_text_for_node(gap["node_b"], nodes_by_id, col)
        concept_result = _propose_concept(node_a, node_b, sample_a, sample_b)

        if not concept_result["has_connection"]:
            print("      -> no substantive connection; noting as rejected", flush=True)
            rejected.append({"gap": gap, "node_a": node_a, "node_b": node_b,
                              "reasoning": concept_result["reasoning"]})
            continue

        print("      -> concept found, designing a test for it...", flush=True)
        design_result = _propose_test_design(concept_result["concept"])
        accepted.append({
            "gap": gap, "node_a": node_a, "node_b": node_b,
            "concept_reasoning": concept_result["reasoning"], "concept": concept_result["concept"],
            "design_reasoning": design_result["reasoning"], "design": design_result["design"],
            "open_questions": design_result["open_questions"],
        })

    return _write_report(accepted, rejected)


def _write_report(accepted: list[dict], rejected: list[dict]) -> Path:
    now = datetime.now(timezone.utc)
    lines = [
        "# LLM-Synthesized Cross-Domain Ideas",
        "",
        f"> **Generated automatically by a local LLM (`{SYNTHESIS_MODEL}`) analyzing gaps in the "
        "memsearch knowledge graph. This is speculative machine synthesis, not verified fact or "
        "human-authored work -- treat every idea below as a lead to evaluate, not a conclusion.**",
        "",
        f"Generated: {now.isoformat()}  ",
        f"Gaps considered: {len(accepted) + len(rejected)} (top by embedding-space gap score)  "
        f"-- {len(accepted)} proposed, {len(rejected)} rejected by the model as not substantive",
        "",
        "---",
        "",
    ]

    has_open_questions = any(item["open_questions"] for item in accepted)
    if has_open_questions:
        lines += [
            "## Summary: open questions to research next",
            "",
            "Everything below was flagged by the model itself as beyond what it can check without "
            "web/literature access. Worth running past Claude (real search) before trusting the "
            "ideas they attach to.",
            "",
        ]
        for i, item in enumerate(accepted, 1):
            if item["open_questions"]:
                lines.append(f"**Idea {i} ({item['node_a']['title']} <-> {item['node_b']['title']}):**")
                lines += [f"- {q}" for q in item["open_questions"]]
                lines.append("")
        lines += ["---", ""]

    for i, item in enumerate(accepted, 1):
        a, b, gap = item["node_a"], item["node_b"], item["gap"]
        lines += [
            f"## Idea {i}: {a['title']} <-> {b['title']}",
            "",
            f"**Areas:** `{a['title']}` ({a['project']}) and `{b['title']}` ({b['project']})  ",
            f"**Gap score:** {gap['gap_score']:.3f} | **Pair similarity:** {gap['pair_similarity']:.3f}",
            "",
            "### Reasoning (concept)",
            item["concept_reasoning"] or "(none captured)",
            "",
            "### Proposed Concept",
            item["concept"],
            "",
            "### Reasoning (test design)",
            item["design_reasoning"] or "(none captured)",
            "",
            "### Test Design",
            item["design"] or "(none captured)",
            "",
            "### Open Questions for External Verification",
            "*Flagged by the model itself as beyond what it can check without web/literature "
            "access -- worth researching (e.g. by asking Claude to look these up) before "
            "treating the idea above as sound.*",
            "",
            *([f"- {q}" for q in item["open_questions"]] if item["open_questions"] else ["(none)"]),
            "",
            "---",
            "",
        ]

    if rejected:
        lines += ["## Considered but rejected by the model", "",
                  "Included for transparency -- these gaps were flagged as related in embedding "
                  "space, but the model concluded on reflection there wasn't a substantive "
                  "connection worth proposing.", ""]
        for item in rejected:
            a, b = item["node_a"], item["node_b"]
            lines += [f"- **{a['title']}** ({a['project']}) <-> **{b['title']}** ({b['project']})"]
        lines.append("")

    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHESIS_DIR / f"synthesis_{now.strftime('%Y%m%d-%H%M%S')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def mine_synthesis_dir() -> None:
    subprocess.run(
        [sys.executable, "-m", "memsearch", "mine", str(SYNTHESIS_DIR), "--project", "synthesis"],
        check=False,
    )
