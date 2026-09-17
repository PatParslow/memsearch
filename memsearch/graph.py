"""Knowledge-graph builder for the memsearch index -- a heaviest-weights
subgraph map over mined content, modelled on parslow.net's own k-blade
concept atlas (tools/site/kblade_atlas.py in the parslow-soft-editorial
repo): top-K cosine similarity graph -> greedy modularity communities ->
force-directed layout -> convex-hull territories -> MST + inter-cluster
bridge edges as the "roads" backbone, plus two embedding-space gap
detectors ported from that repo's tools/site/find_doc_gaps.py.

Node granularity is one node per `source_file` (already the uniform key
across file chunks, PDF chunks, and conversation-session chunks -- see
miner.py / convo_miner.py metadata), with its vector the mean-pooled,
L2-normalized centroid of that file's chunk embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial import ConvexHull

from . import store

DEFAULT_GRAPH_PATH = os.path.expanduser(r"~\.memsearch\graph.json")

CANVAS_W, CANVAS_H, MARGIN = 1600, 1200, 70
TOP_K_NEIGHBORS_DEFAULT = 8
FETCH_BATCH = 5000
KEYWORDS_PER_NODE = 8

PAIR_Z_MIN, PAIR_Z_MAX = 0.5, 2.5
FRONTIER_EXTRAPOLATE = 1.6
MAX_GAPS_PER_KIND = 40
HIERARCHY_FLATTEN_BELOW = 4
HIERARCHY_FANOUT = 8

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_TIMEOUT = 20
LLM_TITLE_SAMPLE_SIZE = 10
LABEL_MAX_DF_FRACTION = 0.08

TERRITORY_COLORS = [
    "#7aa2c9", "#c98a7a", "#8ac97a", "#c9b17a", "#a67ac9",
    "#7ac9b1", "#c97aa6", "#a6c97a", "#7a8ec9", "#c9a67a",
]

_STOPWORDS = set("""
the a an and or but if then so to of in on for with as at by from into
is are was were be been being this that these those it its it's i you
he she they we not no yes do does did done can could should would will
shall may might must have has had having about above after again against
all am any because before below between both down during each few
further here how just more most other over own same some such than too
very what when where which while who whom why your yours yourself
their theirs them there they're we're you're i'm i've you've we've
also like one two three get gets got using used use uses via
doi https http org com www pdf arxiv isbn issn vol pp ed eds
now let's
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")


def _node_id(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]


def _node_title(source_file: str, kind: str) -> str:
    p = Path(source_file)
    if kind == "conversation":
        parent = p.parent.name
        return f"conversation ({parent[:24]}) {p.stem[:8]}"
    return p.stem


def _top_keywords(texts: list[str], k: int = KEYWORDS_PER_NODE) -> list[str]:
    # Count each word at most once per chunk (document frequency within the
    # file), not once per raw occurrence -- otherwise a boilerplate phrase
    # repeated many times in a single chunk (e.g. a citation-placeholder
    # link title stamped on every reference) swamps the real topical words.
    # dict.fromkeys (not a set) for the per-chunk dedup: a bare set's
    # iteration order depends on Python's per-process hash randomization,
    # which made most_common()'s tie-breaking between equal-count words
    # silently non-deterministic across runs (same corpus, different
    # top-3 keywords each `graph build` -- caught by re-running an
    # identical diagnostic twice and getting 110 vs. 68 vs. 0 for the same
    # word). dict.fromkeys preserves first-occurrence order in the text
    # instead, which is deterministic.
    counts = Counter()
    for t in texts:
        words = list(dict.fromkeys(
            w for w in _WORD_RE.findall(t.lower()) if len(w) >= 3 and w not in _STOPWORDS
        ))
        counts.update(words)
    return [w for w, _ in counts.most_common(k)]


def _representative_label(indices, keywords: list[list[str]], global_df: Counter,
                           n_total: int, k: int = 3) -> str:
    """Pick a group/cluster label by IDF-weighted vote among words that
    aren't near-universal across the corpus, not raw frequency. A word that
    shows up in a large fraction of ALL nodes (this corpus's own flagship
    project name, mentioned incidentally even in unrelated files, or a
    boilerplate phrase repeated across many near-duplicate redirect stubs)
    reliably outvotes genuinely distinctive-but-rarer words in any large,
    mixed group -- log-scaled IDF discounting alone isn't a strong enough
    penalty against that (observed empirically: an 11.8%-corpus-wide word
    still won the vote in two DIFFERENT, unrelated branches of the same
    split). A hard max-document-frequency cutoff (same idea as scikit-
    learn's TfidfVectorizer max_df) excludes near-universal words from
    candidacy entirely instead of just discounting them."""
    local_counts = Counter(kw for i in indices for kw in keywords[i][:3])
    if not local_counts:
        return f"{len(indices)} items"
    max_df = LABEL_MAX_DF_FRACTION * n_total
    eligible = {w: c for w, c in local_counts.items() if global_df.get(w, 0) <= max_df}
    if not eligible:
        eligible = local_counts
    scored = {
        w: c * (math.log((n_total + 1) / (global_df.get(w, 0) + 1)) + 1)
        for w, c in eligible.items()
    }
    top = sorted(scored, key=lambda w: scored[w], reverse=True)[:k]
    return ", ".join(top)


def _ollama_available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=2)
        return True
    except Exception:
        return False


def _llm_group_title(sample_titles: list[str], keyword_hint: str) -> str | None:
    """Ask a small local model (via Ollama) for a real natural-language
    cluster title instead of a raw keyword list -- generation is ~0.15-0.2s
    once the model is warm, so this is affordable even across the ~150-200
    group nodes a typical hierarchy build produces. Returns None (falls
    back to the keyword label) on any failure -- this must never break a
    graph build, including the unattended 3am scheduled one."""
    prompt = (
        "These are document/file titles from one cluster of a personal knowledge base:\n"
        + "\n".join(f'- "{t}"' for t in sample_titles)
        + f"\n\nTop shared keywords: {keyword_hint}\n\n"
        "Give a short (2-5 word) descriptive topic label for this cluster. "
        "Reply with ONLY the label, no punctuation, no explanation."
    )
    body = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "keep_alive": "5m",
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_GENERATE_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        title = data.get("response", "").strip().strip('"').strip(".")
        return title or None
    except Exception:
        return None


def _sample_titles(indices, final_nodes: list[dict], k: int = LLM_TITLE_SAMPLE_SIZE) -> list[str]:
    # Evenly spaced across the group rather than the first k in pre_order's
    # DFS order, which would bias toward whichever side of the underlying
    # binary tree happens to be visited first.
    if len(indices) <= k:
        chosen = indices
    else:
        step = max(1, len(indices) // k)
        chosen = indices[::step][:k]
    return [final_nodes[i]["title"] for i in chosen]


def _fetch_scoped_chunks(col, project: str | None, include_code: bool):
    kinds = ["prose", "conversation"] + (["code"] if include_code else [])
    where = {"kind": {"$in": kinds}}
    if project:
        where = {"$and": [where, {"project": project}]}
    offset = 0
    while True:
        batch = col.get(
            limit=FETCH_BATCH, offset=offset, where=where,
            include=["documents", "embeddings", "metadatas"],
        )
        if not batch["ids"]:
            break
        for doc, emb, meta in zip(batch["documents"], batch["embeddings"], batch["metadatas"]):
            yield doc, emb, meta
        offset += FETCH_BATCH


def _build_nodes(col, project: str | None, include_code: bool):
    groups: dict[str, dict] = defaultdict(lambda: {"embs": [], "texts": [], "meta": None})
    for doc, emb, meta in _fetch_scoped_chunks(col, project, include_code):
        sf = meta.get("source_file", "?")
        g = groups[sf]
        g["embs"].append(emb)
        g["texts"].append(doc)
        if g["meta"] is None:
            g["meta"] = meta
    return groups


def _similarity_graph(sim: np.ndarray, top_k: int) -> nx.Graph:
    """Keep only each node's top-K strongest neighbours -- the "heaviest
    weights subgraph" -- ported from
    tools/site/concept_experiments/ga_kblade_graph_communities.py."""
    n = sim.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        row = sim[i]
        neighbor_idx = np.argsort(row)[::-1][:top_k]
        for j in neighbor_idx:
            j = int(j)
            w = float(sim[i, j])
            if w > 0:
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], w)
                else:
                    G.add_edge(i, j, weight=w)
    return G


def _zscore_offdiag(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = mat[mask]
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-9:
        sigma = 1.0
    z = (mat - mu) / sigma
    np.fill_diagonal(z, 0.0)
    return z


def _find_interpolation_gaps(node_ids: list[str], titles: list[str], unit: np.ndarray,
                              top_k: int, top_n: int) -> list[dict]:
    """Pairs that are meaningfully related (z-scored cosine) but not near-
    duplicates, whose midpoint direction has no good third-party match --
    ported from tools/site/find_doc_gaps.py:find_interpolation_gaps."""
    n = len(node_ids)
    if n < 3:
        return []
    sim = unit @ unit.T
    np.fill_diagonal(sim, -1.0)
    z = _zscore_offdiag(sim)

    candidates = []
    seen = set()
    for i in range(n):
        neighbours = np.argsort(sim[i])[::-1][:top_k]
        for j in neighbours:
            j = int(j)
            if not (PAIR_Z_MIN <= z[i, j] <= PAIR_Z_MAX):
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)

            mid = unit[i] + unit[j]
            norm = np.linalg.norm(mid)
            if norm < 1e-9:
                continue
            mid /= norm

            mid_sim_to_pair = float((mid @ unit[key[0]] + mid @ unit[key[1]]) / 2.0)
            mid_sim_all = unit @ mid
            mid_sim_all[key[0]] = -1.0
            mid_sim_all[key[1]] = -1.0
            best_third_idx = int(np.argmax(mid_sim_all))
            best_third_sim = float(mid_sim_all[best_third_idx])

            candidates.append({
                "gap_score": mid_sim_to_pair - best_third_sim,
                "node_a": node_ids[key[0]], "node_b": node_ids[key[1]],
                "title_a": titles[key[0]], "title_b": titles[key[1]],
                "pair_similarity": float(sim[key[0], key[1]]),
                # The bridge/frontier node's own id, not just its title --
                # several files across this corpus share an identical
                # basename (a source copy, a build cache, and a published
                # copy of the same page, for instance), so a bare title is
                # genuinely ambiguous about which physical file was
                # matched. Found via a real report of an apparent "0.91
                # similarity to itself" that turned out to be two
                # different files with the same name, not a bug in the
                # similarity computation.
                "nearest_existing_bridge_node": node_ids[best_third_idx],
                "nearest_existing_bridge": titles[best_third_idx],
                "nearest_existing_bridge_similarity": best_third_sim,
            })

    candidates.sort(key=lambda c: c["gap_score"], reverse=True)
    return candidates[:top_n]


def _find_extrapolation_gaps(node_ids: list[str], titles: list[str], unit: np.ndarray,
                              node_cluster: dict[int, int], clusters_out: list[dict],
                              keywords: list[list[str]], top_n: int) -> list[dict]:
    """Per cluster, project a bit past its own most-outlying member and see
    if anything in the whole corpus is close to that projected point --
    ported from tools/site/find_doc_gaps.py:find_extrapolation_gaps
    (vocabulary-labelling step omitted; frontier node's own keywords stand
    in for a full TF-IDF vocabulary pass)."""
    results = []
    for cluster in clusters_out:
        members = [i for i, ci in node_cluster.items() if ci == cluster["id"]]
        if len(members) < 2:
            continue
        member_vecs = unit[members]
        centroid = member_vecs.mean(axis=0)
        centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-9)

        sims_to_centroid = member_vecs @ centroid_norm
        frontier_idx = members[int(np.argmin(sims_to_centroid))]
        frontier_vec = unit[frontier_idx]

        outward = frontier_vec - centroid_norm
        if np.linalg.norm(outward) < 1e-9:
            continue
        projected = centroid_norm + FRONTIER_EXTRAPOLATE * outward
        projected /= max(np.linalg.norm(projected), 1e-9)

        all_sims = unit @ projected
        all_sims[frontier_idx] = -1.0
        nearest_idx = int(np.argmax(all_sims))
        nearest_sim = float(all_sims[nearest_idx])

        results.append({
            "gap_score": 1.0 - nearest_sim,
            "cluster_id": cluster["id"], "cluster_label": cluster["label"],
            "frontier_node": node_ids[frontier_idx], "frontier_title": titles[frontier_idx],
            "frontier_keywords": keywords[frontier_idx][:4],
            # See the matching comment in _find_interpolation_gaps -- same
            # title-collision ambiguity, same fix (carry the real id).
            "nearest_existing_node": node_ids[nearest_idx],
            "nearest_existing": titles[nearest_idx], "nearest_existing_similarity": nearest_sim,
        })

    results.sort(key=lambda r: r["gap_score"], reverse=True)
    return results[:top_n]


def _bridge_text(node_a: dict, node_b: dict, similarity: float) -> str:
    shared = [k for k in node_a["keywords"] if k in node_b["keywords"]]
    terms = shared[:3] or (node_a["keywords"][:2] + node_b["keywords"][:2])[:3]
    if similarity > 0.75:
        strength = "closely related"
    elif similarity > 0.55:
        strength = "related"
    else:
        strength = "loosely related"
    text = f"{node_a['title']} ({node_a['project']}) is {strength} to {node_b['title']} ({node_b['project']})"
    if terms:
        text += f" -- shared terms: {', '.join(terms)}"
    return text


def _build_hierarchy(unit: np.ndarray, final_nodes: list[dict], keywords: list[list[str]],
                      global_df: Counter, n_total: int, use_llm_titles: bool) -> dict:
    """Real agglomerative clustering (average-linkage, cosine distance) over
    the same per-file mean embeddings the force-directed map uses -- a
    genuine dendrogram, not a fixed Project>Cluster>Document scheme. Subtrees
    at or below HIERARCHY_FLATTEN_BELOW leaves are flattened into one labeled
    group with direct leaf children instead of a long chain of binary splits,
    since a raw dendrogram down to singletons is unusable as a collapsible
    outline."""
    n = unit.shape[0]
    if n == 0:
        return {"kind": "group", "label": "(empty)", "size": 0, "children": []}
    if n == 1:
        return {"kind": "leaf", "node_id": final_nodes[0]["id"]}

    # Ward's method (implicitly Euclidean) rather than average/cosine linkage --
    # on these already-unit-normalized vectors Euclidean distance is a monotonic
    # function of cosine similarity, and average-linkage was observed to chain
    # badly on this corpus (max depth 76, root split 2 vs. 2292) versus Ward's
    # far more balanced result (max depth ~21, root split ~1032 vs. 1262).
    Z = linkage(unit, method="ward")
    root = to_tree(Z, rd=False)

    def label_for(indices) -> str:
        return _representative_label(indices, keywords, global_df, n_total)

    title_count = 0

    def llm_title_for(indices, kw_label: str) -> str | None:
        nonlocal title_count
        if not use_llm_titles:
            return None
        title_count += 1
        if title_count % 20 == 0:
            print(f"    ...{title_count} group titles generated", flush=True)
        return _llm_group_title(_sample_titles(indices, final_nodes), kw_label)

    def split_into_fanout(node, target_fanout: int) -> list:
        # scipy's linkage/to_tree is inherently a strictly-binary dendrogram
        # (every merge combines exactly 2 clusters). Two children per level
        # gives almost no differentiation at a glance -- repeatedly peeling
        # the largest remaining piece into its own 2 linkage-children turns
        # the same real clustering into a wider, more legible display tree
        # (up to target_fanout children) without inventing new cluster math.
        active = [node]
        while len(active) < target_fanout:
            splittable = [(i, a) for i, a in enumerate(active) if not a.is_leaf()]
            if not splittable:
                break
            i, biggest = max(splittable, key=lambda ia: len(ia[1].pre_order()))
            active[i : i + 1] = [biggest.get_left(), biggest.get_right()]
        return active

    def walk(node) -> dict:
        if node.is_leaf():
            return {"kind": "leaf", "node_id": final_nodes[node.id]["id"]}
        leaf_idx = node.pre_order(lambda x: x.id)
        kw_label = label_for(leaf_idx)
        if len(leaf_idx) <= HIERARCHY_FLATTEN_BELOW:
            return {
                "kind": "group", "label": kw_label, "llm_title": llm_title_for(leaf_idx, kw_label),
                "size": len(leaf_idx),
                "children": [{"kind": "leaf", "node_id": final_nodes[i]["id"]} for i in leaf_idx],
            }
        children = split_into_fanout(node, HIERARCHY_FANOUT)
        return {
            "kind": "group", "label": kw_label, "llm_title": llm_title_for(leaf_idx, kw_label),
            "size": len(leaf_idx),
            "children": [walk(c) for c in children],
        }

    return walk(root)


def build_graph(project: str | None = None, top_k: int = TOP_K_NEIGHBORS_DEFAULT,
                 include_code: bool = False, store_path: str = store.DEFAULT_STORE_PATH) -> dict:
    col = store.get_collection(store_path)
    groups = _build_nodes(col, project, include_code)
    node_ids_key = list(groups.keys())  # source_file per group, stable ordering
    n = len(node_ids_key)
    if n == 0:
        return {"generated_at": datetime.now(timezone.utc).isoformat(),
                "scope": {"project": project, "top_k": top_k, "include_code": include_code},
                "nodes": [], "clusters": [], "roads": []}

    dim = len(groups[node_ids_key[0]]["embs"][0])
    vecs = np.zeros((n, dim), dtype=np.float32)
    keywords: list[list[str]] = []
    metas = []
    for i, sf in enumerate(node_ids_key):
        g = groups[sf]
        vecs[i] = np.mean(np.array(g["embs"], dtype=np.float32), axis=0)
        keywords.append(_top_keywords(g["texts"]))
        metas.append(g["meta"])

    global_df: Counter = Counter()
    for kws in keywords:
        global_df.update(set(kws[:3]))

    use_llm_titles = _ollama_available()
    print(f"  -> [memsearch graph] LLM group titles via Ollama: "
          f"{'enabled (' + OLLAMA_MODEL + ')' if use_llm_titles else 'unavailable, using keyword labels only'}",
          flush=True)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1e-9
    unit = vecs / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)

    G = _similarity_graph(sim, min(top_k, max(1, n - 1)))
    base_clusters = [sorted(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")]

    pos = nx.spring_layout(G, weight="weight", seed=1337, iterations=100)
    coords = np.array([pos[i] for i in range(n)])

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    canvas = np.zeros_like(coords)
    canvas[:, 0] = MARGIN + (coords[:, 0] - lo[0]) / span[0] * (CANVAS_W - 2 * MARGIN)
    canvas[:, 1] = MARGIN + (1.0 - (coords[:, 1] - lo[1]) / span[1]) * (CANVAS_H - 2 * MARGIN)

    for members in base_clusters:
        pts = canvas[members]
        centroid = pts.mean(axis=0)
        scale = 1.0 + 0.35 * np.sqrt(max(0, len(members) - 8))
        canvas[members] = centroid + (pts - centroid) * scale
    lo2, hi2 = canvas.min(axis=0), canvas.max(axis=0)
    span2 = np.maximum(hi2 - lo2, 1e-6)
    canvas[:, 0] = MARGIN + (canvas[:, 0] - lo2[0]) / span2[0] * (CANVAS_W - 2 * MARGIN)
    canvas[:, 1] = MARGIN + (canvas[:, 1] - lo2[1]) / span2[1] * (CANVAS_H - 2 * MARGIN)

    node_cluster = {i: ci for ci, members in enumerate(base_clusters) for i in members}

    clusters_out = []
    for ci, members in enumerate(base_clusters):
        pts = canvas[members]
        label = _representative_label(members, keywords, global_df, n) or f"cluster {ci}"
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hv = pts[hull.vertices]
        else:
            c = pts.mean(axis=0)
            hv = np.array([c + [dx, dy] for dx, dy in [(-24, -24), (24, -24), (24, 24), (-24, 24)]])
        centroid = pts.mean(axis=0)
        clusters_out.append({
            "id": ci, "label": label,
            "hull": [[round(float(x), 1), round(float(y), 1)] for x, y in hv],
            "centroid": [round(float(centroid[0]), 1), round(float(centroid[1]), 1)],
            "color": TERRITORY_COLORS[ci % len(TERRITORY_COLORS)],
            "size": len(members),
        })

    final_nodes = []
    for i, sf in enumerate(node_ids_key):
        meta = metas[i]
        x, y = canvas[i]
        final_nodes.append({
            "id": _node_id(sf), "source_file": sf,
            "title": _node_title(sf, meta.get("kind", "prose")),
            "project": meta.get("project", "?"), "category": meta.get("category", "?"),
            "kind": meta.get("kind", "prose"),
            "cluster": node_cluster.get(i, -1),
            "x": round(float(x), 1), "y": round(float(y), 1),
            "keywords": keywords[i], "n_chunks": len(groups[sf]["embs"]),
        })
    nodes_by_idx = final_nodes  # same order as node_ids_key

    if use_llm_titles:
        for ci, c in enumerate(clusters_out):
            members = base_clusters[ci]
            c["llm_title"] = _llm_group_title(_sample_titles(members, final_nodes), c["label"])

    G_geo = nx.Graph()
    for u, v in G.edges():
        G_geo.add_edge(u, v, dist=float(np.hypot(*(canvas[u] - canvas[v]))))
    if G_geo.number_of_edges() > 0:
        mst = nx.minimum_spanning_tree(G_geo, weight="dist")
        road_edges = set(tuple(sorted(e)) for e in mst.edges())
    else:
        road_edges = set()

    best_bridge = {}
    for u, v in G.edges():
        cu, cv = node_cluster.get(u), node_cluster.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        key = tuple(sorted((cu, cv)))
        d = float(np.hypot(*(canvas[u] - canvas[v])))
        if key not in best_bridge or d < best_bridge[key][2]:
            best_bridge[key] = (u, v, d)
    for u, v, _ in best_bridge.values():
        road_edges.add(tuple(sorted((u, v))))

    roads = []
    for u, v in road_edges:
        nA, nB = nodes_by_idx[u], nodes_by_idx[v]
        roads.append({
            "from": nA["id"], "to": nB["id"],
            "similarity": round(float(sim[u, v]), 3),
            "bridge_text": _bridge_text(nA, nB, float(sim[u, v])),
        })

    print(f"  -> [memsearch graph] Detecting gaps ({n} nodes)...", flush=True)
    interp_gaps = _find_interpolation_gaps(
        [nd["id"] for nd in final_nodes], [nd["title"] for nd in final_nodes],
        unit, min(top_k, max(1, n - 1)), MAX_GAPS_PER_KIND)
    extrap_gaps = _find_extrapolation_gaps(
        [nd["id"] for nd in final_nodes], [nd["title"] for nd in final_nodes],
        unit, node_cluster, clusters_out, keywords, MAX_GAPS_PER_KIND)

    print(f"  -> [memsearch graph] Building hierarchy ({n} nodes)...", flush=True)
    hierarchy = _build_hierarchy(unit, final_nodes, keywords, global_df, n, use_llm_titles)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"project": project, "top_k": top_k, "include_code": include_code},
        "canvas": {"width": CANVAS_W, "height": CANVAS_H},
        "nodes": final_nodes,
        "clusters": clusters_out,
        "roads": roads,
        "interpolation_gaps": interp_gaps,
        "extrapolation_gaps": extrap_gaps,
        "hierarchy": hierarchy,
    }


def save_graph(data: dict, path: str = DEFAULT_GRAPH_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path: str = DEFAULT_GRAPH_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
