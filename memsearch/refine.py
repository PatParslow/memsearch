"""Iterative refine-then-reverify loop for synthesis ideas whose
verification found a real contradiction.

verify.py's job stops at reporting findings. This module closes the
loop: take the concept, design, and findings, produce a revised
proposal that corrects mischaracterizations and reframes "novel
mechanism" claims as "known technique applied to a new domain" where
that's what the evidence actually shows (a real, if narrower,
contribution -- not a failure), then re-verify the revision. Repeats
until no more "contradicted" findings remain or a max-iteration cap is
hit.

The refine step can run on either model:
- Claude ("claude"): already has the real research grounding it just
  produced verifying the idea; a text-reasoning refine call with no
  forced new search is cheap relative to the verify call that preceded
  it (WebSearch stays available if the revision raises something new
  worth checking).
- The local model ("local"): no API turn at all, but has to be handed
  the findings as plain text since it didn't do the research itself.
- "auto" (default): try Claude first, fall back to the local model if
  Claude fails for any reason. This matters in practice, not just in
  theory -- Claude's own usage-session limit was hit mid-session during
  real testing of this pipeline, and the loop should degrade to
  continue working, not die, when that happens.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import synthesis
from . import verify

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "changelog": {"type": "string", "description": "1-3 sentences: what changed and why, based on the verification findings"},
        "revised_concept": {"type": "string", "description": "The updated 3-5 sentence concept, correcting mischaracterizations and reframing claims the findings contradicted or reframed"},
        "revised_what_to_build": {"type": "string"},
        "revised_success_metrics": {"type": "string"},
        "revised_how_to_falsify": {"type": "string"},
        "new_open_questions": {
            "type": "array", "items": {"type": "string"},
            "description": "New or still-unresolved concrete, checkable claims the revised idea now depends on -- empty array if the revision fully resolves prior concerns with nothing new to check",
        },
    },
    "required": [
        "changelog", "revised_concept", "revised_what_to_build",
        "revised_success_metrics", "revised_how_to_falsify", "new_open_questions",
    ],
}

MAX_ITERATIONS_DEFAULT = 3


def _build_refine_prompt(concept: str, design: str, findings: list[dict]) -> str:
    findings_text = "\n".join(
        f"- Q: {f['question']}\n  Verdict: {f['verdict']}\n  {f['summary']}"
        for f in findings
    )
    return f"""A speculative idea was proposed for a personal knowledge base, then checked against real research. Some checks found problems.

ORIGINAL CONCEPT: {concept}

ORIGINAL TEST DESIGN: {design}

VERIFICATION FINDINGS:
{findings_text}

Revise the idea into a sharper, more defensible version:
- Correct any mischaracterization the research exposed (a "reframed" verdict means the framing was wrong but a corrected version of the claim still holds -- use the correction, don't discard the idea).
- Where a "precedented" verdict found a similar approach already exists elsewhere (e.g. an established technique like Dreamer-style world models), do NOT simply abandon the idea for lacking novelty -- assess whether applying that same mechanism specifically in THIS knowledge base's own domain is still a worthwhile, narrower experiment worth running.
- Only actually weaken or narrow a claim where a "contradicted" verdict shows it's genuinely wrong, and consider whether a different parameter, scope, or step size (not the core mechanism) would sidestep the specific failure found, rather than concluding the whole approach is dead.
- If nothing meaningful can be salvaged, say so plainly in the changelog rather than forcing a revision that doesn't actually address the findings."""


def _is_refine_result_valid(r: dict) -> bool:
    fields = ("changelog", "revised_concept", "revised_what_to_build", "revised_success_metrics", "revised_how_to_falsify")
    return all(len(str(r.get(f, "")).split()) >= 5 for f in fields) and isinstance(r.get("new_open_questions"), list)


def _refine_via_local(concept: str, design: str, findings: list[dict]) -> dict | None:
    prompt = _build_refine_prompt(concept, design, findings)
    return synthesis._ollama_generate_structured(prompt, REFINE_SCHEMA, _is_refine_result_valid)


def _refine_via_claude(concept: str, design: str, findings: list[dict]) -> tuple[dict | None, float]:
    prompt = _build_refine_prompt(concept, design, findings)
    result, cost = verify._call_claude_structured(prompt, REFINE_SCHEMA)
    if result and not _is_refine_result_valid(result):
        result = None
    return result, cost


def _refine(concept: str, design: str, findings: list[dict], refiner: str) -> tuple[dict | None, float, str]:
    """Returns (result, cost, model_actually_used)."""
    if refiner in ("claude", "auto"):
        result, cost = _refine_via_claude(concept, design, findings)
        if result:
            return result, cost, "claude"
        if refiner == "claude":
            return None, cost, "claude"
        print("      ! Claude refine unavailable, falling back to local model", flush=True)
    result = _refine_via_local(concept, design, findings)
    return result, 0.0, "local"


def _extract_idea_block(report_text: str, idea_number: int) -> dict | None:
    """Pulls the current concept/design/open-questions for one specific
    idea, regardless of whether it's already been verified once -- unlike
    verify._parse_ideas, which deliberately SKIPS already-verified ideas
    (that function answers "what still needs a first verify pass"; this
    one answers "what is idea N's current state", which the refine loop
    needs regardless of prior verification)."""
    headings = list(re.finditer(r"(?m)^## Idea (\d+): ", report_text))
    for idx, m in enumerate(headings):
        if int(m.group(1)) != idea_number:
            continue
        start = m.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(report_text)
        block = report_text[start:end]
        title = block.split("\n", 1)[0].strip()
        concept_m = re.search(r"### Proposed Concept\s*\n(.*?)(?=\n### )", block, re.DOTALL)
        design_m = re.search(r"### Test Design\s*\n(.*?)(?=\n### )", block, re.DOTALL)
        oq_m = re.search(
            r"### Open Questions for External Verification\s*\n\*.*?\*\s*\n\n(.*?)(?=\n---|\n### |\Z)",
            block, re.DOTALL,
        )
        questions = []
        if oq_m:
            questions = [
                line.lstrip("-").strip() for line in oq_m.group(1).splitlines()
                if line.strip().startswith("-")
            ]
        return {
            "title": title,
            "concept": concept_m.group(1).strip() if concept_m else "",
            "design": design_m.group(1).strip() if design_m else "",
            "questions": questions,
        }
    return None


def _strip_prior_flat_verification(report_text: str, idea_number: int) -> str:
    """If this idea already went through a single-pass `memsearch verify`
    (the flat, non-iterating predecessor of this loop) before ever
    running the loop on it, its block has one "### External Verification
    Results" section with no iteration number. Remove it before writing
    the iteration log, so the report doesn't show that flat section
    AND a redundant "Iteration 1" covering the identical (cache-hit)
    findings."""
    heading = f"## Idea {idea_number}: "
    heading_pos = report_text.find(heading)
    if heading_pos == -1:
        return report_text
    next_m = re.search(r"(?m)^## Idea \d+: ", report_text[heading_pos + len(heading):])
    block_end = heading_pos + len(heading) + next_m.start() if next_m else len(report_text)
    section_m = re.search(
        r"\n\n### External Verification Results \(via Claude, real web search\)\n\n.*?(?=\n---|\n### |\Z)",
        report_text[heading_pos:block_end], re.DOTALL,
    )
    if not section_m:
        return report_text
    abs_start = heading_pos + section_m.start()
    abs_end = heading_pos + section_m.end()
    return report_text[:abs_start] + report_text[abs_end:]


def _write_iteration_log(path: Path, idea_number: int, iteration_log: list[dict]) -> None:
    text = path.read_text(encoding="utf-8")
    text = _strip_prior_flat_verification(text, idea_number)

    heading = f"## Idea {idea_number}: "
    heading_pos = text.find(heading)
    if heading_pos == -1:
        return
    next_m = re.search(r"(?m)^## Idea \d+: ", text[heading_pos + len(heading):])
    block_end = heading_pos + len(heading) + next_m.start() if next_m else len(text)

    insertion = "\n"
    for entry in iteration_log:
        n = entry["iteration"]
        insertion += f"\n### Verification (Iteration {n}, via Claude, real web search)\n\n"
        insertion += verify._render_findings(entry["findings"])
        r = entry.get("refinement")
        if r:
            insertion += f"\n### Refinement (Iteration {n}, via {r['model']})\n\n"
            insertion += f"**Changelog:** {r['changelog']}\n\n"
            insertion += f"**Revised Concept:** {r['revised_concept']}\n\n"
            insertion += (
                "**Revised Test Design:**\n"
                f"- What to build: {r['revised_what_to_build']}\n"
                f"- Success metrics: {r['revised_success_metrics']}\n"
                f"- How to falsify it: {r['revised_how_to_falsify']}\n\n"
            )
            if r.get("new_open_questions"):
                insertion += "**New Open Questions:**\n" + "\n".join(f"- {q}" for q in r["new_open_questions"]) + "\n"

    search_region = text[heading_pos:block_end]
    sep_pos = search_region.rfind("\n---")
    insert_at = heading_pos + sep_pos if sep_pos != -1 else block_end
    new_text = text[:insert_at] + insertion + text[insert_at:]
    path.write_text(new_text, encoding="utf-8")


def run_refine_loop(
    report_path: str, idea_number: int,
    max_iterations: int = MAX_ITERATIONS_DEFAULT, refiner: str = "auto",
    max_cost: float = 5.0,
) -> tuple[Path, float, str]:
    path = Path(report_path)
    text = path.read_text(encoding="utf-8")
    state = _extract_idea_block(text, idea_number)
    if not state:
        raise RuntimeError(f"Idea {idea_number} not found in {report_path}")
    if not state["questions"]:
        raise RuntimeError(f"Idea {idea_number} has no open questions to verify -- nothing for the loop to check")

    cache = verify._load_json(verify.VERIFY_CACHE_PATH)
    if not isinstance(cache, dict):
        cache = {}
    index = verify._load_json(verify.SYNTHESIS_INDEX_PATH)
    if not isinstance(index, list):
        index = []

    concept, design, questions = state["concept"], state["design"], state["questions"]
    total_cost = 0.0
    iteration_log: list[dict] = []
    final_status = "accepted"
    final_findings: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        if total_cost >= max_cost:
            print(f"  Iteration {iteration}: skipping -- budget cap (${max_cost:.2f}) reached", flush=True)
            break
        print(f"  Iteration {iteration}: verifying...", flush=True)
        result, cost, was_cached = verify._verify_idea(concept, questions, cache)
        total_cost += cost
        verify._save_json(verify.VERIFY_CACHE_PATH, cache)
        cache_note = " (cached)" if was_cached else ""
        print(f"    -> cost: ${cost:.4f} (running total: ${total_cost:.4f}){cache_note}", flush=True)
        if not result or not result.get("findings"):
            print("    ! no usable verification result, stopping loop", flush=True)
            break

        findings = result["findings"]
        final_findings = findings
        status = verify._derive_status(findings)
        final_status = status
        iteration_log.append({"iteration": iteration, "findings": findings, "refinement": None})

        if status == "accepted":
            print(f"    -> no contradictions, converged after {iteration} iteration(s)", flush=True)
            break
        if iteration == max_iterations:
            print(f"    -> still has contradictions after {max_iterations} iteration(s), stopping", flush=True)
            break

        print(f"    -> contradicted finding(s) present, refining (refiner={refiner})...", flush=True)
        refined, refine_cost, used_model = _refine(concept, design, findings, refiner)
        total_cost += refine_cost
        print(f"    -> refine cost: ${refine_cost:.4f} (running total: ${total_cost:.4f}, used {used_model})", flush=True)
        if not refined:
            print("    ! refine failed, stopping loop with current state", flush=True)
            break

        iteration_log[-1]["refinement"] = {"model": used_model, **refined}
        concept = refined["revised_concept"]
        design = (
            f"- **What to build:** {refined['revised_what_to_build']}\n"
            f"- **Success metrics:** {refined['revised_success_metrics']}\n"
            f"- **How to falsify it:** {refined['revised_how_to_falsify']}"
        )
        questions = refined["new_open_questions"] or questions

    if not iteration_log:
        # No iteration ever produced real findings (e.g. the very first
        # verify call failed -- content policy, session limit). Leaving
        # the index entry untouched here matters: final_status defaults
        # to "accepted" and would otherwise overwrite whatever real
        # verdicts/status already existed with an empty, falsely-clean
        # result -- a failed check must never look like a passed one.
        print("  ! No successful verification this run -- leaving prior "
              "status/verdicts in the index untouched", flush=True)
        return path, total_cost, "unchanged"

    _write_iteration_log(path, idea_number, iteration_log)

    verdict_tally: dict[str, int] = {}
    for f in final_findings:
        v = f.get("verdict", "unresolved")
        verdict_tally[v] = verdict_tally.get(v, 0) + 1
    report_name = path.name
    for entry in index:
        if entry.get("report_file") == report_name and entry.get("idea_number") == idea_number:
            entry["status"] = final_status
            entry["verdicts"] = verdict_tally
            entry["iterations"] = len(iteration_log)
            break
    verify._save_json(verify.SYNTHESIS_INDEX_PATH, index)

    return path, total_cost, final_status
