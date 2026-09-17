# memsearch

A personal local semantic-search and knowledge-graph tool over one user's
own work: project files across a set of tracked project directories, every
Claude Code conversation transcript (including subagent runs), and a
local PDF library. It embeds everything into a local ChromaDB store, builds
a force-directed map and a clustered mindmap over it, and layers an
experimental LLM-driven idea-synthesis pipeline on top: proposing
cross-domain ideas from gaps in the embedding space, verifying them against
external sources via headless Claude, refining them iteratively, scanning
for synergies between them, and rolling related ideas up into (or breaking
a large one down into) named projects.

## Status: work in progress, not a research tool

This is an active experiment in tool-building, not a validated research
method. The idea-synthesis pipeline (`synthesize`, `verify`, `refine`,
`synergy`, `consolidate`, `decompose`) uses local and hosted LLMs to
propose, critique, and reconcile connections between disparate material.
That is genuinely useful for surfacing leads worth a human's attention, and
worth continuing to build on -- but its output is speculative machine
synthesis, not verified fact, and it has not been evaluated against any
standard that would make it a sound basis for actual research conclusions.
Treat everything it produces as something to investigate, not something to
cite.

## What's here

- **Mining**: `memsearch mine <dir>` and `memsearch mine-convos` chunk and
  embed project files and Claude Code transcripts respectively, skipping
  unchanged files on repeat runs.
- **Search**: `memsearch search "<query>"` runs semantic search across
  everything mined, optionally scoped to one project.
- **Graph**: `memsearch graph build` computes a knowledge graph (nearest
  neighbours, interpolation gaps between distant clusters) and
  `memsearch graph serve` serves an interactive local web UI over it --
  force-directed map, clustered mindmap, and a kanban-style board for
  synthesized ideas and projects.
- **Idea synthesis** (experimental, local LLM via Ollama):
  `memsearch synthesize` proposes cross-domain ideas from the graph's own
  gaps; `memsearch verify` checks a report's open questions against real
  web search via headless Claude; `memsearch refine` iterates an idea
  against its own verification findings; `memsearch synergy` scans live
  ideas for shared techniques or dependencies; `memsearch consolidate`
  rolls related ideas into a named parent project; `memsearch decompose`
  breaks one idea down into well-scoped sub-projects with explicit
  interfaces, and can reconcile multiple independent decomposition
  attempts of the same idea into one.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally, with a model available for
  the synthesis/synergy/decompose pipeline (see `SYNTHESIS_MODEL` in
  `memsearch/synthesis.py`). Development has used Qwen3 8B at a 3-bit
  (`iq3xxs`) quantization specifically to fit inside 16GB of VRAM -- a
  full-precision or larger model would likely improve output quality, at
  the cost of needing more capable hardware.
- The `claude` CLI, logged in, for `memsearch verify`'s headless web-search
  checks (draws from your Claude subscription usage allowance, not a
  separate API key)

## Install

```
pip install -e .
```

## Usage

```
memsearch mine <project-dir>
memsearch mine-convos
memsearch search "<query>"
memsearch graph build
memsearch graph serve
```

Run `memsearch <command> --help` for each command's options.
