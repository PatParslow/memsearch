"""memsearch CLI -- mine projects and conversations, search across them."""

from __future__ import annotations

import argparse
import sys

from . import convo_miner, graph, miner, refine as refine_mod, store, synergy, synthesis, verify as verify_mod
from .gpu import enable_gpu


def cmd_mine(args):
    exclude = set(args.exclude.split(",")) if args.exclude else None
    stats = miner.mine_project(
        args.dir, project=args.project, dry_run=args.dry_run, exclude_dirs=exclude
    )
    print(f"\nProcessed {stats['processed']} files, {stats['chunks']} chunks filed")
    print(f"Unchanged (skipped): {stats['skipped_unchanged']}")
    print(f"Binary/non-text (skipped): {stats['skipped_binary']}")


def cmd_mine_convos(args):
    stats = convo_miner.mine_convos(dry_run=args.dry_run)
    print(f"\nProcessed {stats['sessions_processed']} sessions, {stats['chunks']} chunks filed")
    print(f"Unchanged (skipped): {stats['sessions_unchanged']}")


def cmd_search(args):
    col = store.get_collection()
    where = None
    if args.project:
        where = {"project": args.project}
    hits = store.search(col, args.query, n_results=args.limit, where=where)
    if not hits:
        print(f'\nNo results for: "{args.query}"')
        return
    print(f'\nResults for: "{args.query}"\n')
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['project']} / {h['category']}")
        print(f"    Source: {h['source_file']}")
        print(f"    Match:  {h['similarity']}\n")
        for line in h["text"].strip().split("\n"):
            print(f"    {line}")
        print()


def cmd_graph_build(args):
    data = graph.build_graph(project=args.project, top_k=args.top_k, include_code=args.include_code)
    graph.save_graph(data)

    from . import annotations
    conn = annotations.get_db()
    auto_gaps = []
    for g in data.get("interpolation_gaps", []):
        desc = (f"{g['title_a']} and {g['title_b']} are related (similarity {g['pair_similarity']:.2f}) "
                f"but nothing sits between them -- nearest existing bridge is "
                f"\"{g['nearest_existing_bridge']}\" ({g['nearest_existing_bridge_similarity']:.2f})")
        auto_gaps.append({"node_a": g["node_a"], "node_b": g["node_b"], "description": desc, "score": g["gap_score"]})
    for g in data.get("extrapolation_gaps", []):
        desc = (f"Beyond \"{g['frontier_title']}\" (edge of \"{g['cluster_label']}\"), nothing covers that "
                f"territory -- nearest existing content is \"{g['nearest_existing']}\" "
                f"({g['nearest_existing_similarity']:.2f})")
        auto_gaps.append({"node_a": g["frontier_node"], "node_b": None, "description": desc, "score": g["gap_score"]})
    annotations.upsert_auto_gaps(conn, auto_gaps)
    conn.close()

    print(f"\nGraph built: {len(data['nodes'])} nodes, {len(data['clusters'])} clusters, {len(data['roads'])} roads")
    print(f"Gaps detected: {len(data.get('interpolation_gaps', []))} interpolation, "
          f"{len(data.get('extrapolation_gaps', []))} extrapolation")
    print(f"Saved to {graph.DEFAULT_GRAPH_PATH}")


def cmd_graph_serve(args):
    from . import webui
    webui.serve(port=args.port, open_browser=not args.no_browser)


def cmd_prune(args):
    stats = store.prune_missing()
    print(f"\nChecked {stats['files_checked']} distinct source files")
    print(f"Missing on disk: {stats['files_missing']}")
    print(f"Chunks deleted: {stats['chunks_deleted']}")


def cmd_unmine(args):
    n = store.delete_path_prefix(args.path)
    print(f"\nDeleted {n} chunks under {args.path}")


def cmd_synthesize(args):
    print(f"\nSynthesizing cross-domain ideas from gaps {args.offset+1}-{args.offset+args.limit} "
          f"(ranked by gap score)...")
    out_path = synthesis.run_synthesis(limit=args.limit, offset=args.offset)
    print(f"\nReport written to {out_path}")
    if not args.no_mine:
        print("Mining the report into the 'synthesis' project...")
        synthesis.mine_synthesis_dir()


def cmd_verify(args):
    print(f"\nVerifying open questions in {args.report} via Claude (real web search, "
          f"budget cap ${args.max_cost:.2f})...")
    out_path, total_cost = verify_mod.verify_report(args.report, max_cost=args.max_cost)
    print(f"\nUpdated {out_path} -- total cost ${total_cost:.4f}")
    if not args.no_mine:
        print("Re-mining the report...")
        synthesis.mine_synthesis_dir()


def cmd_refine(args):
    print(f"\nRefine loop on idea {args.idea} in {args.report} "
          f"(refiner={args.refiner}, max {args.max_iterations} iteration(s), budget cap ${args.max_cost:.2f})...")
    out_path, total_cost, status = refine_mod.run_refine_loop(
        args.report, args.idea, max_iterations=args.max_iterations,
        refiner=args.refiner, max_cost=args.max_cost,
    )
    print(f"\nUpdated {out_path} -- final status: {status} -- total cost ${total_cost:.4f}")
    if not args.no_mine:
        print("Re-mining the report...")
        synthesis.mine_synthesis_dir()


def cmd_synergy(args):
    print(f"\nScanning for synergies across all live ideas "
          f"(min similarity {args.min_similarity:.2f}, top {args.top_n})...")
    out_path = synergy.run_synergy_scan(min_similarity=args.min_similarity, top_n=args.top_n)
    print(f"\nWritten to {out_path}")


def cmd_consolidate(args):
    children = []
    for spec in args.children:
        report_file, _, idea_str = spec.rpartition(":")
        if not report_file or not idea_str.isdigit():
            raise SystemExit(f"Invalid child spec '{spec}' -- expected report_file.md:idea_number")
        children.append((report_file, int(idea_str)))
    print(f"\nConsolidating {len(children)} idea(s) into a parent project...")
    project = synergy.consolidate_ideas(children)
    print(f"\nCreated project {project['id']}: {project['title']}")
    print(project["description"])


def cmd_status(args):
    breakdown = store.status_breakdown()
    total = sum(sum(cats.values()) for cats in breakdown.values())
    print(f"\nmemsearch status -- {total} chunks\n")
    for project, cats in sorted(breakdown.items()):
        print(f"  PROJECT: {project}")
        for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat:20} {count:6}")
        print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    enable_gpu()

    parser = argparse.ArgumentParser(prog="memsearch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mine = sub.add_parser("mine", help="Mine a project directory")
    p_mine.add_argument("dir")
    p_mine.add_argument("--project", default=None, help="Project name (default: dir name)")
    p_mine.add_argument(
        "--exclude", default=None,
        help="Comma-separated subdirectory names to skip (in addition to the default noise list)",
    )
    p_mine.add_argument("--dry-run", action="store_true")
    p_mine.set_defaults(func=cmd_mine)

    p_convos = sub.add_parser("mine-convos", help="Mine Claude Code conversation transcripts")
    p_convos.add_argument("--dry-run", action="store_true")
    p_convos.set_defaults(func=cmd_mine_convos)

    p_search = sub.add_parser("search", help="Semantic search across mined content")
    p_search.add_argument("query")
    p_search.add_argument("--project", default=None)
    p_search.add_argument("--limit", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="Show what's been mined")
    p_status.set_defaults(func=cmd_status)

    p_synth = sub.add_parser(
        "synthesize", help="Propose cross-domain ideas from the graph's own interpolation gaps"
    )
    p_synth.add_argument("--limit", type=int, default=8, help="Number of gaps to process (default: 8)")
    p_synth.add_argument("--offset", type=int, default=0, help="Skip the top N gaps (for paging past already-processed ones)")
    p_synth.add_argument("--no-mine", action="store_true", help="Don't auto-mine the report afterward")
    p_synth.set_defaults(func=cmd_synthesize)

    p_prune = sub.add_parser("prune", help="Delete chunks for files that no longer exist on disk")
    p_prune.set_defaults(func=cmd_prune)

    p_unmine = sub.add_parser("unmine", help="Delete all chunks whose source_file starts with a given path")
    p_unmine.add_argument("path")
    p_unmine.set_defaults(func=cmd_unmine)

    p_verify = sub.add_parser(
        "verify", help="Research a synthesis report's open questions via headless Claude (real web search)"
    )
    p_verify.add_argument("report", help="Path to a synthesis_*.md report file")
    p_verify.add_argument("--max-cost", type=float, default=5.0, help="Stop after this much spend, in USD (default: 5.00)")
    p_verify.add_argument("--no-mine", action="store_true", help="Don't re-mine the report afterward")
    p_verify.set_defaults(func=cmd_verify)

    p_refine = sub.add_parser(
        "refine", help="Iteratively refine + re-verify one idea until it stops contradicting itself"
    )
    p_refine.add_argument("report", help="Path to a synthesis_*.md report file")
    p_refine.add_argument("idea", type=int, help="Idea number within the report")
    p_refine.add_argument("--max-iterations", type=int, default=refine_mod.MAX_ITERATIONS_DEFAULT)
    p_refine.add_argument("--refiner", choices=["claude", "local", "auto"], default="auto",
                           help="Which model performs the refine step (default: auto -- Claude, falling back to local)")
    p_refine.add_argument("--max-cost", type=float, default=5.0, help="Stop after this much spend, in USD (default: 5.00)")
    p_refine.add_argument("--no-mine", action="store_true", help="Don't re-mine the report afterward")
    p_refine.set_defaults(func=cmd_refine)

    p_synergy = sub.add_parser(
        "synergy", help="Scan all live synthesis ideas for shared techniques, dependencies, or synergies"
    )
    p_synergy.add_argument("--min-similarity", type=float, default=synergy.MIN_SIMILARITY_DEFAULT,
                            help=f"Minimum cosine similarity to consider a pair (default: {synergy.MIN_SIMILARITY_DEFAULT})")
    p_synergy.add_argument("--top-n", type=int, default=synergy.TOP_N_DEFAULT,
                            help=f"Max pairs to characterize with the local model (default: {synergy.TOP_N_DEFAULT})")
    p_synergy.set_defaults(func=cmd_synergy)

    p_consolidate = sub.add_parser(
        "consolidate", help="Roll up related ideas into a named parent project"
    )
    p_consolidate.add_argument("children", nargs="+", help="report_file.md:idea_number, one per related idea")
    p_consolidate.set_defaults(func=cmd_consolidate)

    p_graph = sub.add_parser("graph", help="Build/serve the knowledge-graph map")
    graph_sub = p_graph.add_subparsers(dest="graph_command", required=True)

    p_gbuild = graph_sub.add_parser("build", help="Build graph.json + detect gaps")
    p_gbuild.add_argument("--project", default=None, help="Scope the graph to one project (default: everything)")
    p_gbuild.add_argument("--top-k", type=int, default=8, help="Neighbors kept per node (default: 8)")
    p_gbuild.add_argument("--include-code", action="store_true", help="Include code-kind files as nodes too")
    p_gbuild.set_defaults(func=cmd_graph_build)

    p_gserve = graph_sub.add_parser("serve", help="Serve the interactive graph UI")
    p_gserve.add_argument("--port", type=int, default=8765)
    p_gserve.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    p_gserve.set_defaults(func=cmd_graph_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
