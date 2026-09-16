(() => {
  "use strict";

  const svg = document.getElementById("map");
  const banner = document.getElementById("hover-banner");
  const statusLine = document.getElementById("status-line");

  let graphData = null;
  let nodesById = new Map();
  let transform = { x: 0, y: 0, scale: 1 };
  let selectedNodeId = null;
  let linkMode = false;
  let linkPending = null;

  const filters = { search: "", project: "", prose: true, conversation: true, code: false };

  const NODE_FONT_BASE = 10;
  const TERRITORY_FONT_BASE = 13;

  // ---- data loading ----

  async function loadGraph() {
    statusLine.textContent = "loading...";
    const res = await fetch("/api/graph");
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      statusLine.textContent = err.error || "failed to load graph";
      return;
    }
    graphData = await res.json();
    nodesById = new Map(graphData.nodes.map((n) => [n.id, n]));
    populateProjectFilter();
    statusLine.textContent = `${graphData.nodes.length} nodes, ${graphData.clusters.length} clusters, ${graphData.roads.length} roads`;
    fitView();
    render();
  }

  function populateProjectFilter() {
    const sel = document.getElementById("project-filter");
    const projects = [...new Set(graphData.nodes.map((n) => n.project))].sort();
    for (const p of projects) {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      sel.appendChild(opt);
    }
  }

  function nodeVisible(n) {
    if (filters.project && n.project !== filters.project) return false;
    if (n.kind === "prose" && !filters.prose) return false;
    if (n.kind === "conversation" && !filters.conversation) return false;
    if (n.kind === "code" && !filters.code) return false;
    if (filters.search) {
      const s = filters.search.toLowerCase();
      const hay = (n.title + " " + n.keywords.join(" ")).toLowerCase();
      if (!hay.includes(s)) return false;
    }
    return true;
  }

  // ---- rendering ----

  function el(tag, attrs, ns) {
    const e = document.createElementNS(ns || "http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    return e;
  }

  function fitView() {
    transform = { x: 40, y: 40, scale: 1 };
  }

  function applyTransform() {
    const g = document.getElementById("viewport");
    if (g) g.setAttribute("transform", `translate(${transform.x},${transform.y}) scale(${transform.scale})`);
  }

  function render() {
    if (!graphData) return;
    svg.innerHTML = "";
    const viewport = el("g", { id: "viewport" });
    svg.appendChild(viewport);

    const visible = new Set(graphData.nodes.filter(nodeVisible).map((n) => n.id));

    // territories
    for (const c of graphData.clusters) {
      const pts = c.hull.map((p) => p.join(",")).join(" ");
      viewport.appendChild(el("polygon", {
        class: "territory-hull", points: pts, fill: c.color, stroke: c.color,
      }));
      const label = el("text", {
        class: "territory-label", x: c.centroid[0], y: c.centroid[1],
        "font-size": TERRITORY_FONT_BASE,
      });
      label.textContent = c.label;
      viewport.appendChild(label);
    }

    // roads
    for (const r of graphData.roads) {
      const a = nodesById.get(r.from), b = nodesById.get(r.to);
      if (!a || !b || !visible.has(a.id) || !visible.has(b.id)) continue;
      const line = el("line", { class: "road-line", x1: a.x, y1: a.y, x2: b.x, y2: b.y });
      line.addEventListener("pointerenter", () => showBanner(r.bridge_text));
      line.addEventListener("pointerleave", hideBanner);
      viewport.appendChild(line);
    }

    // manual links
    for (const link of (graphData.manual_links || [])) {
      const a = nodesById.get(link.node_a), b = nodesById.get(link.node_b);
      if (!a || !b) continue;
      const line = el("line", { class: "manual-link-line", x1: a.x, y1: a.y, x2: b.x, y2: b.y });
      const label = link.label ? `${a.title} ↔ ${b.title}: ${link.label}` : `${a.title} ↔ ${b.title}`;
      line.addEventListener("pointerenter", () => showBanner(label));
      line.addEventListener("pointerleave", hideBanner);
      viewport.appendChild(line);
    }

    // interpolation gaps -- ghost line between the two nodes with nothing bridging them
    const gapStatus = new Map((graphData.gap_status || []).map((g) => [g.id, g.status]));
    for (const g of (graphData.interpolation_gaps || [])) {
      if (gapStatusFor(gapStatus, "auto", g.node_a, g.node_b, describeInterp(g)) === "dismissed") continue;
      const a = nodesById.get(g.node_a), b = nodesById.get(g.node_b);
      if (!a || !b || !visible.has(a.id) || !visible.has(b.id)) continue;
      const line = el("line", { class: "gap-line", x1: a.x, y1: a.y, x2: b.x, y2: b.y });
      line.addEventListener("pointerenter", () => showBanner("Possible gap: " + describeInterp(g)));
      line.addEventListener("pointerleave", hideBanner);
      viewport.appendChild(line);
    }

    // extrapolation gaps -- marker projected outward from the frontier node
    for (const g of (graphData.extrapolation_gaps || [])) {
      if (gapStatusFor(gapStatus, "auto", g.frontier_node, null, describeExtrap(g)) === "dismissed") continue;
      const frontier = nodesById.get(g.frontier_node);
      if (!frontier || !visible.has(frontier.id)) continue;
      const cluster = graphData.clusters.find((c) => c.id === g.cluster_id);
      let dx = 30, dy = -30;
      if (cluster) {
        dx = frontier.x - cluster.centroid[0];
        dy = frontier.y - cluster.centroid[1];
        const len = Math.hypot(dx, dy) || 1;
        dx = (dx / len) * 34; dy = (dy / len) * 34;
      }
      const mx = frontier.x + dx, my = frontier.y + dy;
      viewport.appendChild(el("line", { class: "gap-line", x1: frontier.x, y1: frontier.y, x2: mx, y2: my }));
      const marker = el("circle", { class: "gap-marker", cx: mx, cy: my, r: 5, "data-base-r": 5 });
      marker.addEventListener("pointerenter", () => showBanner("Possible gap: " + describeExtrap(g)));
      marker.addEventListener("pointerleave", hideBanner);
      viewport.appendChild(marker);
    }

    // nodes
    for (const n of graphData.nodes) {
      const group = el("g", { class: "atlas-node", "data-id": n.id });
      if (!visible.has(n.id)) group.classList.add("dimmed");
      if (n.id === selectedNodeId) group.classList.add("selected");
      if (linkPending === n.id) group.classList.add("link-pending");
      const r = 4 + Math.min(10, Math.log2(1 + n.n_chunks));
      const cluster = graphData.clusters.find((c) => c.id === n.cluster);
      const circle = el("circle", { cx: n.x, cy: n.y, r, "data-base-r": r, fill: cluster ? cluster.color : "#888" });
      const label = el("text", { x: n.x + r + 3, y: n.y + 3, "font-size": NODE_FONT_BASE });
      label.textContent = n.title;
      group.appendChild(circle);
      group.appendChild(label);
      group.addEventListener("pointerenter", () => { group.classList.add("hovered"); showBanner(`${n.title} (${n.project} / ${n.category})`); });
      group.addEventListener("pointerleave", () => { group.classList.remove("hovered"); hideBanner(); });
      group.addEventListener("click", (ev) => { ev.stopPropagation(); onNodeClick(n); });
      viewport.appendChild(group);
    }

    applyTransform();
    rescaleForZoom();
    renderGapsTab();
  }

  // Node/label/marker sizes are stored as map-space base values (data-base-r,
  // the font-size set at render time) and counter-scaled here against the
  // current zoom so they stay a constant size on screen -- otherwise they'd
  // balloon along with everything else under the viewport's scale transform.
  // Stroke widths instead use CSS vector-effect:non-scaling-stroke, which
  // the SVG spec handles natively with no JS needed.
  function rescaleForZoom() {
    const s = transform.scale;
    for (const c of svg.querySelectorAll("circle[data-base-r]")) {
      c.setAttribute("r", (parseFloat(c.dataset.baseR) / s).toFixed(2));
    }
    for (const t of svg.querySelectorAll(".atlas-node text")) {
      t.setAttribute("font-size", (NODE_FONT_BASE / s).toFixed(2));
    }
    for (const t of svg.querySelectorAll(".territory-label")) {
      t.setAttribute("font-size", (TERRITORY_FONT_BASE / s).toFixed(2));
    }
  }

  function describeInterp(g) {
    return `${g.title_a} and ${g.title_b} are related (similarity ${g.pair_similarity.toFixed(2)}) but nothing sits between them -- nearest existing bridge is "${g.nearest_existing_bridge}" (${g.nearest_existing_bridge_similarity.toFixed(2)})`;
  }
  function describeExtrap(g) {
    return `Beyond "${g.frontier_title}" (edge of "${g.cluster_label}"), nothing covers that territory -- nearest existing content is "${g.nearest_existing}" (${g.nearest_existing_similarity.toFixed(2)})`;
  }
  function gapStatusFor(map, kind, nodeA, nodeB, description) {
    // mirrors annotations.gap_id()'s hashing input shape server-side; client
    // only needs to know if a manually-dismissed row already covers this
    // gap, so we match by (node_a, node_b, description) linearly instead --
    // small lists, no need to reimplement the sha256 id client-side.
    for (const g of (graphData.gap_status || [])) {
      if (g.kind === kind && g.node_a === nodeA && (g.node_b || null) === (nodeB || null) && g.description === description) {
        return g.status;
      }
    }
    return "open";
  }

  function showBanner(text) {
    banner.textContent = text;
    banner.style.display = "block";
  }
  function hideBanner() {
    banner.style.display = "none";
  }

  // ---- node selection / notes ----

  async function onNodeClick(n) {
    if (linkMode) {
      if (!linkPending) {
        linkPending = n.id;
        render();
        return;
      }
      if (linkPending === n.id) {
        linkPending = null;
        render();
        return;
      }
      // No window.prompt() here -- it's a blocking modal that freezes the
      // whole page; links start unlabeled and can be described via a note
      // on either endpoint node instead.
      await fetch("/api/links", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_a: linkPending, node_b: n.id, label: null }),
      });
      linkPending = null;
      await loadGraph();
      return;
    }
    selectedNodeId = n.id;
    render();
    await showNodeDetail(n);
  }

  async function showNodeDetail(n) {
    document.getElementById("node-empty").style.display = "none";
    const detail = document.getElementById("node-detail");
    detail.style.display = "block";
    document.getElementById("nd-title").textContent = n.title;
    document.getElementById("nd-meta").textContent = `${n.project} / ${n.category} · ${n.kind} · ${n.n_chunks} chunk(s)`;
    document.getElementById("nd-keywords").textContent = n.keywords.join(", ");
    const openLink = document.getElementById("nd-open-source");
    openLink.href = `/api/source?path=${encodeURIComponent(n.source_file)}`;
    openLink.title = n.source_file;

    const notesRes = await fetch(`/api/notes?node_id=${encodeURIComponent(n.id)}`);
    const notes = await notesRes.json();
    const notesDiv = document.getElementById("nd-notes");
    notesDiv.innerHTML = "";
    for (const note of notes) {
      const item = document.createElement("div");
      item.className = "note-item";
      item.textContent = note.text;
      const date = document.createElement("div");
      date.className = "note-date";
      date.textContent = new Date(note.created_at * 1000).toLocaleString();
      item.appendChild(date);
      notesDiv.appendChild(item);
    }
    document.getElementById("nd-note-text").value = "";
  }

  document.getElementById("nd-note-save").addEventListener("click", async () => {
    if (!selectedNodeId) return;
    const textArea = document.getElementById("nd-note-text");
    const text = textArea.value.trim();
    if (!text) return;
    await fetch("/api/notes", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: selectedNodeId, text }),
    });
    await showNodeDetail(nodesById.get(selectedNodeId));
  });

  // ---- gaps tab ----

  function renderGapsTab() {
    const list = document.getElementById("gaps-list");
    list.innerHTML = "";
    const rows = [];
    for (const g of (graphData.interpolation_gaps || [])) {
      rows.push({ kind: "interpolation", desc: describeInterp(g), nodeA: g.node_a, nodeB: g.node_b });
    }
    for (const g of (graphData.extrapolation_gaps || [])) {
      rows.push({ kind: "extrapolation", desc: describeExtrap(g), nodeA: g.frontier_node, nodeB: null });
    }
    for (const g of (graphData.gap_status || [])) {
      if (g.kind === "manual") rows.push({ kind: "manual", desc: g.description, nodeA: g.node_a, nodeB: g.node_b, id: g.id, status: g.status });
    }

    const gapStatus = new Map((graphData.gap_status || []).map((g) => [g.id, g.status]));
    for (const row of rows) {
      const status = row.status || gapStatusFor(gapStatus, row.kind === "manual" ? "manual" : "auto", row.nodeA, row.nodeB, row.desc);
      if (status === "dismissed") continue;
      const item = document.createElement("div");
      item.className = "gap-item";
      const kindLabel = document.createElement("div");
      kindLabel.className = "gap-kind";
      kindLabel.textContent = row.kind;
      const desc = document.createElement("div");
      desc.textContent = row.desc;
      const btn = document.createElement("button");
      btn.textContent = "Dismiss";
      btn.addEventListener("click", () => dismissGap(row));
      item.append(kindLabel, desc, btn);
      list.appendChild(item);
    }
    if (!list.children.length) {
      list.innerHTML = '<p class="muted">No open gaps.</p>';
    }
  }

  async function dismissGap(row) {
    if (row.id) {
      await fetch(`/api/gaps/${row.id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "dismissed" }),
      });
    } else {
      // Auto gap never persisted until dismissed -- create it pre-dismissed,
      // under kind:"auto" so the id matches what the next `graph build`
      // upsert computes (same gap_id() hashing on the server), otherwise
      // the dismissal would land on a different row and not stick.
      await fetch("/api/gaps", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_a: row.nodeA, node_b: row.nodeB, description: row.desc, kind: "auto", status: "dismissed" }),
      });
    }
    await loadGraph();
  }

  // ---- pan / zoom ----

  let dragging = false, dragStart = null;

  svg.addEventListener("pointerdown", (ev) => {
    dragging = true;
    dragStart = { x: ev.clientX - transform.x, y: ev.clientY - transform.y };
    svg.classList.add("dragging");
  });
  window.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    transform.x = ev.clientX - dragStart.x;
    transform.y = ev.clientY - dragStart.y;
    applyTransform();
  });
  window.addEventListener("pointerup", () => {
    dragging = false;
    svg.classList.remove("dragging");
  });
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
    const factor = ev.deltaY < 0 ? 1.1 : 0.9;
    const preX = (cx - transform.x) / transform.scale;
    const preY = (cy - transform.y) / transform.scale;
    transform.scale = Math.max(0.15, Math.min(6, transform.scale * factor));
    transform.x = cx - preX * transform.scale;
    transform.y = cy - preY * transform.scale;
    applyTransform();
    rescaleForZoom();
  }, { passive: false });

  // ---- controls ----

  document.getElementById("search-box").addEventListener("input", (ev) => {
    filters.search = ev.target.value;
    render();
  });
  document.getElementById("project-filter").addEventListener("change", (ev) => {
    filters.project = ev.target.value;
    render();
  });
  document.getElementById("kind-prose").addEventListener("change", (ev) => { filters.prose = ev.target.checked; render(); });
  document.getElementById("kind-conversation").addEventListener("change", (ev) => { filters.conversation = ev.target.checked; render(); });
  document.getElementById("kind-code").addEventListener("change", (ev) => { filters.code = ev.target.checked; render(); });

  document.getElementById("link-mode-btn").addEventListener("click", (ev) => {
    linkMode = !linkMode;
    linkPending = null;
    ev.target.classList.toggle("active", linkMode);
    render();
  });

  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".tab-btn")) b.classList.remove("active");
      for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  }

  loadGraph();
})();
