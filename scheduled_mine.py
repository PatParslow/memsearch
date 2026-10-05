"""Wrapper for Windows Task Scheduler: incrementally re-mine every tracked
project plus all Claude Code conversations. Safe to run daily -- already-
current files are skipped via content-hash comparison, so a re-run only
does real work on what's actually changed.

Also checks aisi.gov.uk for any new research papers or blog posts before
mining (see aisi_source.py) and saves them under F:\\books\\aisi-research,
so they're picked up by the ordinary F:\\books mine call right after --
this keeps that corpus current without a separate schedule.

Writes its own timestamped log under logs/ -- the scheduled task invokes
this script directly with no shell redirection, so without this the only
way to see what happened was having a terminal window open by chance.

Note: subprocess output is written directly to the log file (not also
echoed to the console) -- a child process's stdout writes to the real OS
file descriptor, which reassigning sys.stdout in this process does NOT
intercept, so the file has to be the subprocess's actual stdout target,
not something layered on top of a Python-level redirect."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from memsearch import aisi_source

SYNTHESIS_DIR = Path(__file__).resolve().parent / "synthesis"

PROJECT_DIRS = [
    r"D:\Projects_Organized\parslow-soft-editorial",
    r"D:\Projects_Organized\parslow-site-editor",
    r"D:\avernelle\avernelle",
    r"D:\echoes_of_the_ice",
    r"D:\mudpy",
    r"D:\multiverse_VM",
    r"D:\oll",
    r"D:\terrain",
    r"D:\Universal",
    r"E:\globe",
    r"E:\tierra-clone",
    r"D:\PatLang",
    r"F:\storyline",
    r"F:\papers",
    r"F:\PatLang-PDF",
    r"F:\books",
]

# Per-directory --exclude value, for subdirectories that are dev-scratch
# noise rather than real content (see memsearch-graph-and-prune memory:
# a batch of ~600 AI-drafted, never-reviewed articles with unverified
# placeholder citations was found dominating cluster labels sitewide).
EXCLUDE_DIRS = {
    r"D:\Projects_Organized\parslow-soft-editorial": "overnight_runs",
}

LOG_DIR = Path(__file__).parent / "logs"


def run(args: list[str], log_file) -> int:
    header = f"\n=== memsearch {' '.join(args)} ===\n"
    print(header, end="", flush=True)
    log_file.write(header)
    log_file.flush()
    result = subprocess.run(
        [sys.executable, "-m", "memsearch", *args],
        stdout=log_file, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        msg = f"  -> exited with code {result.returncode}\n"
        print(msg, end="", flush=True)
        log_file.write(msg)
    return result.returncode


def newest_report() -> Path | None:
    reports = sorted(SYNTHESIS_DIR.glob("synthesis_*.md"), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


def idea_numbers(report: Path) -> list[int]:
    return [int(n) for n in re.findall(r"^## Idea (\d+):", report.read_text(encoding="utf-8"), flags=re.MULTILINE)]


def run_synthesis(log_file) -> list[str]:
    """Synthesis after the indexing: propose ideas from the graph's gaps, refine each one with the
    local model (no paid calls), mine the finished report once, then scan for synergies."""
    failures: list[str] = []
    if run(["synthesize", "--no-mine"], log_file) != 0:
        return ["synthesize"]
    report = newest_report()
    if report is None:
        return ["synthesize (no report written)"]
    for n in idea_numbers(report):
        if run(["refine", str(report), str(n), "--refiner", "local", "--max-cost", "1.00", "--no-mine"], log_file) != 0:
            failures.append(f"refine {n}")
    if run(["mine", str(SYNTHESIS_DIR), "--project", "synthesis"], log_file) != 0:
        failures.append("mine synthesis")
    if run(["synergy"], log_file) != 0:
        failures.append("synergy")
    return failures


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    print(f"Logging to {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        start_msg = f"memsearch scheduled run -- {datetime.now().isoformat()}\n"
        print(start_msg, end="", flush=True)
        log_file.write(start_msg)

        failures = []

        aisi_header = "\n=== fetch-aisi (check for new AISI papers/blog posts) ===\n"
        print(aisi_header, end="", flush=True)
        log_file.write(aisi_header)
        log_file.flush()
        aisi_result = aisi_source.fetch_new()
        if aisi_result["error"]:
            msg = f"  -> {aisi_result['error']}\n"
            failures.append("fetch-aisi")
        else:
            msg = (f"  -> {aisi_result['new_papers']} new paper(s), {aisi_result['new_blogs']} new blog post(s), "
                   f"{aisi_result['excluded_organisation']} Organisation post(s) excluded\n")
        print(msg, end="", flush=True)
        log_file.write(msg)
        for line in aisi_result["log"]:
            log_file.write(f"  {line}\n")
        log_file.flush()

        for d in PROJECT_DIRS:
            args = ["mine", d]
            if d in EXCLUDE_DIRS:
                args += ["--exclude", EXCLUDE_DIRS[d]]
            if run(args, log_file) != 0:
                failures.append(d)
        if run(["mine-convos"], log_file) != 0:
            failures.append("mine-convos")
        if run(["prune"], log_file) != 0:
            failures.append("prune")
        if run(["graph", "build"], log_file) != 0:
            failures.append("graph build")

        failures.extend(run_synthesis(log_file))

        if failures:
            done_msg = f"\nDone, with {len(failures)} failure(s): {failures}\n"
        else:
            done_msg = "\nDone, no failures.\n"
        print(done_msg, end="", flush=True)
        log_file.write(done_msg)


if __name__ == "__main__":
    main()
