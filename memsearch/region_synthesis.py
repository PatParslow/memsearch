"""LLM-driven exploration of blade-defined extrapolation regions (see
blade.py and graph.py's _find_extrapolation_gaps) -- the generalization
of synthesis.py's pairwise interpolation-gap synthesis to a region rather
than a single point. Three modes, runnable separately or together for
the same region, per the discussion that motivated this:

- point: treat each of the region's sampled directions' nearest real
  neighbour as an ordinary two-area connection, reusing synthesis.py's
  existing concept/test-design machinery completely unchanged.
- region: describe the blade's spanning region qualitatively -- which
  members define its boundary, and in what direction it's spreading --
  and ask, openly, whether a plausible new concept sits somewhere past
  that edge. No concrete anchor point; more room to reason about the
  region as a whole, more room to produce something ungrounded.
- combined: the region prompt, but grounded with the concrete nearest-
  neighbour points the point-samples already found, named as reference
  points inside the region rather than left to reason about the region
  alone.

region and combined share one function (`_propose_region_concept`) --
combined is literally region called with `reference_points` filled in,
not a separate prompt/schema, since the only real difference is whether
concrete anchors are named.

Deliberately does NOT try to make region/combined ideas fully
refine/verify-loop-compatible (refine.py's _extract_idea_block expects a
"### Test Design" heading it uses to extract build-idea design text) --
these are open-ended extrapolative leads, not scoped build-or-writing
ideas, and forcing that shape would misrepresent what they are. They do
use the same "## Idea N: ..." / "### Proposed Concept" convention so
synergy.py's cross-idea scan (which only reads the concept text) picks
them up automatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import graph as graph_mod
from . import store, synthesis

REGION_SYNTHESIS_DIR = synthesis.SYNTHESIS_DIR
REGION_INDEX_PATH = synthesis.SYNTHESIS_INDEX_PATH

REGION_CONCEPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step reasoning about whether a genuine, plausible new concept exists "
                           "past this region's edge, in the direction its boundary members are already spreading",
        },
        "has_plausible_concept": {"type": "boolean"},
        "proposed_concept": {
            "type": "string",
            "description": "A 3-5 sentence concrete idea for what might exist past this edge. Empty string if has_plausible_concept is false.",
        },
        "idea_type": {
            "type": "string", "enum": ["build", "writing"],
            "description": "'build' if the extrapolated idea is a technique, system, or mechanism worth "
                           "implementing and testing. 'writing' if its real value is conceptual/pedagogical.",
        },
    },
    "required": ["reasoning", "has_plausible_concept", "proposed_concept", "idea_type"],
}


def _region_boundary_details(gap: dict, nodes_by_id: dict, col) -> list[dict]:
    details = []
    for node_id, title in zip(gap["region_boundary_nodes"], gap["region_boundary_titles"]):
        node = nodes_by_id.get(node_id)
        project = node["project"] if node else "?"
        sample = synthesis._sample_text_for_node(node_id, nodes_by_id, col)
        details.append({"node_id": node_id, "title": title, "project": project, "sample": sample})
    return details


def _build_region_prompt(cluster_label: str, boundary: list[dict], reference_points: list[dict] | None) -> str:
    boundary_block = "\n\n".join(
        f'- "{b["title"]}" (project: {b["project"]})\n  {b["sample"][:800]}' for b in boundary
    )
    reference_block = ""
    if reference_points:
        lines = "\n".join(
            f'- "{r["title"]}" (project: {r["project"]}, similarity to this region {r["similarity"]:.2f})'
            for r in reference_points
        )
        reference_block = (
            "\n\nThe closest existing content actually found in several different directions past this "
            f"edge (concrete reference points -- these already exist in the corpus, they are not part of "
            f"the boundary above):\n{lines}\n"
        )
    return f"""You are analyzing a personal knowledge base's cluster of related material, "{cluster_label}", to see whether something genuinely new might exist just beyond its current edge.

These documents mark the outer boundary of the cluster -- the area is spreading toward them, but nothing further out in that direction has been mined yet:

{boundary_block}
{reference_block}
Think about what these boundary documents have in common in the direction they're spreading, and whether a genuine, plausible new concept exists a step further out along that same direction -- not a summary of what's already here, and not a forced connection to something unrelated. Be honest if nothing plausible suggests itself; concluding there's no real extrapolation here is a legitimate and useful answer, not a failure to find one.

If a concept exists, decide whether it's a 'build' idea (a technique, system, or mechanism worth implementing and testing) or a 'writing' idea (a genuine conceptual piece worth explaining to a reader)."""


def _propose_region_concept(cluster_label: str, boundary: list[dict], reference_points: list[dict] | None = None) -> dict:
    prompt = _build_region_prompt(cluster_label, boundary, reference_points)

    def is_valid(r: dict) -> bool:
        if not str(r.get("reasoning", "")).strip():
            return False
        if not r.get("has_plausible_concept"):
            return True
        return bool(r.get("proposed_concept", "").strip())

    result = synthesis._ollama_generate_structured(prompt, REGION_CONCEPT_SCHEMA, is_valid)
    if not result:
        return {"reasoning": None, "concept": None, "has_connection": False, "idea_type": "build"}
    has_connection = bool(result.get("has_plausible_concept")) and bool(result.get("proposed_concept", "").strip())
    return {
        "reasoning": result.get("reasoning"),
        "concept": result.get("proposed_concept") if has_connection else None,
        "has_connection": has_connection,
        "idea_type": result.get("idea_type") if result.get("idea_type") in ("build", "writing") else "build",
    }


def _group_regions(graph_data: dict, limit: int) -> list[dict]:
    """One entry per distinct cluster (not per sample) -- a region's
    several point-samples share the same boundary, so region/combined
    mode needs the boundary described once, not once per sample."""
    gaps = sorted(graph_data.get("extrapolation_gaps", []), key=lambda g: g["gap_score"], reverse=True)
    by_cluster: dict[int, list[dict]] = {}
    order: list[int] = []
    for g in gaps:
        cid = g["cluster_id"]
        if cid not in by_cluster:
            by_cluster[cid] = []
            order.append(cid)
        by_cluster[cid].append(g)
    return [{"cluster_id": cid, "cluster_label": by_cluster[cid][0]["cluster_label"], "samples": by_cluster[cid]}
            for cid in order[:limit]]


def run_region_synthesis(limit: int = 8, mode: str = "all", graph_path: str = graph_mod.DEFAULT_GRAPH_PATH) -> Path:
    if mode not in ("point", "region", "combined", "all"):
        raise ValueError(f"Unknown mode {mode!r} -- expected point, region, combined, or all")

    graph_data = graph_mod.load_graph(graph_path)
    nodes_by_id = {n["id"]: n for n in graph_data["nodes"]}
    regions = _group_regions(graph_data, limit)
    if not regions:
        raise RuntimeError("No extrapolation gaps found in the graph -- run `memsearch graph build` first")

    col = store.get_collection()
    point_accepted: list[dict] = []
    point_rejected: list[dict] = []
    region_ideas: list[dict] = []  # covers both "region" and "combined" modes

    run_point = mode in ("point", "all")
    run_region = mode in ("region", "all")
    run_combined = mode in ("combined", "all")

    for i, region in enumerate(regions, 1):
        cluster_label = region["cluster_label"]
        samples = region["samples"]
        print(f"  [{i}/{len(regions)}] region \"{cluster_label}\" ({len(samples)} sample(s))...", flush=True)
        boundary = _region_boundary_details(samples[0], nodes_by_id, col)

        if run_point:
            for s, gap in enumerate(samples, 1):
                frontier = nodes_by_id.get(gap["frontier_node"])
                nearest = nodes_by_id.get(gap["nearest_existing_node"])
                if not frontier or not nearest:
                    continue
                sample_a = synthesis._sample_text_for_node(gap["frontier_node"], nodes_by_id, col)
                sample_b = synthesis._sample_text_for_node(gap["nearest_existing_node"], nodes_by_id, col)
                print(f"      point {s}/{len(samples)}: {frontier['title'][:30]} -> {nearest['title'][:30]}...", flush=True)
                concept_result = synthesis._propose_concept(frontier, nearest, sample_a, sample_b)
                if not concept_result["has_connection"]:
                    point_rejected.append({"gap": gap, "node_a": frontier, "node_b": nearest,
                                            "reasoning": concept_result["reasoning"]})
                    continue
                idea_type = concept_result["idea_type"]
                if idea_type == "writing":
                    design_result = synthesis._propose_writing_design(frontier, nearest, sample_a, sample_b, concept_result["concept"])
                else:
                    design_result = synthesis._propose_test_design(frontier, nearest, sample_a, sample_b, concept_result["concept"])
                point_accepted.append({
                    "gap": gap, "node_a": frontier, "node_b": nearest, "idea_type": idea_type,
                    "concept_reasoning": concept_result["reasoning"], "concept": concept_result["concept"],
                    "design_reasoning": design_result["reasoning"], "design": design_result["design"],
                    "open_questions": design_result["open_questions"],
                })

        if run_region:
            print("      region (no anchors)...", flush=True)
            result = _propose_region_concept(cluster_label, boundary, reference_points=None)
            region_ideas.append({"kind": "region", "cluster_label": cluster_label, "boundary": boundary, **result})

        if run_combined:
            reference_points = [
                {"title": g["nearest_existing"],
                 "project": (nodes_by_id.get(g["nearest_existing_node"]) or {}).get("project", "?"),
                 "similarity": g["nearest_existing_similarity"]}
                for g in samples
            ]
            print("      combined (with anchors)...", flush=True)
            result = _propose_region_concept(cluster_label, boundary, reference_points=reference_points)
            region_ideas.append({"kind": "combined", "cluster_label": cluster_label, "boundary": boundary,
                                  "reference_points": reference_points, **result})

    report_path = _write_report(point_accepted, point_rejected, region_ideas)
    _update_index(report_path, point_accepted, point_rejected, region_ideas)
    return report_path


def _write_report(point_accepted: list[dict], point_rejected: list[dict], region_ideas: list[dict]) -> Path:
    now = datetime.now(timezone.utc)
    lines = [
        "# LLM-Synthesized Extrapolation-Region Ideas",
        "",
        f"> **Generated automatically by a local LLM (`{synthesis.SYNTHESIS_MODEL}`) exploring blade-defined "
        "regions beyond the memsearch knowledge graph's own clusters. This is speculative machine synthesis, "
        "not verified fact or human-authored work -- treat every idea below as a lead to evaluate, not a conclusion.**",
        "",
        f"Generated: {now.isoformat()}  ",
        f"Point ideas: {len(point_accepted)} proposed, {len(point_rejected)} rejected as not substantive  "
        f"-- Region/combined ideas: {len(region_ideas)} considered",
        "",
        "---",
        "",
    ]

    idea_number = 0
    for item in point_accepted:
        idea_number += 1
        a, b, gap = item["node_a"], item["node_b"], item["gap"]
        idea_type = item.get("idea_type", "build")
        design_heading = "Writing Plan" if idea_type == "writing" else "Test Design"
        lines += [
            f"## Idea {idea_number}: {a['title']} <-> {b['title']} (point)",
            "",
            f"**Type:** {idea_type} | **Areas:** `{a['title']}` ({a['project']}) and `{b['title']}` ({b['project']})  ",
            f"**Gap score:** {gap['gap_score']:.3f} | **Blade rank:** {gap['blade_rank']}",
            "",
            "### Proposed Concept",
            item["concept"],
            "",
            "### Reasoning (concept)",
            item["concept_reasoning"] or "(none captured)",
            "",
            f"### {design_heading}",
            item["design"] or "(none captured)",
            "",
            f"### Reasoning ({design_heading.lower()})",
            item["design_reasoning"] or "(none captured)",
            "",
            "### Open Questions for External Verification",
            "*Flagged by the model itself as beyond what it can check without web/literature "
            "access -- worth researching before treating the idea above as sound.*",
            "",
            *([f"- {q}" for q in item["open_questions"]] if item["open_questions"] else ["(none)"]),
            "",
            "---",
            "",
        ]

    for item in region_ideas:
        if not item.get("has_connection"):
            continue
        idea_number += 1
        boundary_names = ", ".join(f'"{b["title"]}"' for b in item["boundary"])
        lines += [
            f"## Idea {idea_number}: Beyond {item['cluster_label']} ({item['kind']})",
            "",
            f"**Type:** {item['idea_type']} | **Region boundary:** {boundary_names}",
        ]
        if item["kind"] == "combined":
            refs = ", ".join(f'"{r["title"]}" ({r["similarity"]:.2f})' for r in item.get("reference_points", []))
            lines.append(f"**Reference points used:** {refs}")
        lines += [
            "",
            "### Proposed Concept",
            item["concept"],
            "",
            "### Reasoning",
            item["reasoning"] or "(none captured)",
            "",
            "---",
            "",
        ]

    if point_rejected or any(not item.get("has_connection") for item in region_ideas):
        lines += ["## Considered but rejected by the model", "",
                  "Included for transparency -- these regions/pairs were flagged as related in embedding "
                  "space, but the model concluded on reflection there wasn't a substantive concept worth proposing.", ""]
        for item in point_rejected:
            a, b = item["node_a"], item["node_b"]
            lines += [
                f'- **{a["title"]}** ({a["project"]}) <-> **{b["title"]}** ({b["project"]}) (point)',
                f"  {item['reasoning'] or '(no reasoning captured)'}",
                "",
            ]
        for item in region_ideas:
            if item.get("has_connection"):
                continue
            lines += [
                f'- **Beyond {item["cluster_label"]}** ({item["kind"]})',
                f"  {item['reasoning'] or '(no reasoning captured)'}",
                "",
            ]

    REGION_SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REGION_SYNTHESIS_DIR / f"region_synthesis_{now.strftime('%Y%m%d-%H%M%S')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _update_index(report_path: Path, point_accepted: list[dict], point_rejected: list[dict], region_ideas: list[dict]) -> None:
    entries = []
    if REGION_INDEX_PATH.exists():
        try:
            entries = json.loads(REGION_INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []

    report_abs_path = str(report_path.resolve())
    report_name = report_path.name
    idea_number = 0
    for item in point_accepted:
        idea_number += 1
        entries.append({
            "node_a": item["gap"]["frontier_node"], "node_b": item["gap"]["nearest_existing_node"],
            "report_path": report_abs_path, "report_file": report_name,
            "idea_number": idea_number, "accepted": True, "status": "accepted",
            "title": f"{item['node_a']['title']} <-> {item['node_b']['title']} (point)",
            "idea_type": item.get("idea_type", "build"),
        })
    for item in point_rejected:
        entries.append({
            "node_a": item["gap"]["frontier_node"], "node_b": item["gap"]["nearest_existing_node"],
            "report_path": report_abs_path, "report_file": report_name,
            "idea_number": None, "accepted": False, "status": "rejected",
            "title": f"{item['node_a']['title']} <-> {item['node_b']['title']} (point)",
        })
    for item in region_ideas:
        if not item.get("has_connection"):
            entries.append({
                "node_a": None, "node_b": None,
                "report_path": report_abs_path, "report_file": report_name,
                "idea_number": None, "accepted": False, "status": "rejected",
                "title": f"Beyond {item['cluster_label']} ({item['kind']})",
            })
            continue
        idea_number += 1
        entries.append({
            "node_a": None, "node_b": None,
            "report_path": report_abs_path, "report_file": report_name,
            "idea_number": idea_number, "accepted": True, "status": "accepted",
            "title": f"Beyond {item['cluster_label']} ({item['kind']})",
            "idea_type": item.get("idea_type", "build"),
        })

    REGION_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGION_INDEX_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
