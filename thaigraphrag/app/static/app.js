/* Thai GraphRAG demo UI.
 * No framework, no CDN — every page calls the real REST API on this origin. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`);
  return body;
};

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (n) => (n === null || n === undefined || Number.isNaN(n)) ? "–"
  : (typeof n === "number" ? n.toLocaleString("en-US") : n);
const fx = (n, d = 3) => (n === null || n === undefined || Number.isNaN(Number(n)))
  ? "–" : Number(n).toFixed(d);
const err = (m) => `<div class="err">${esc(m)}</div>`;

let STATS = null;
const LABEL_TH = {};

/* ── navigation ──────────────────────────────────────────────────────── */

const loaders = {};
let loaded = {};

function show(page) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  location.hash = page;
  if (loaders[page] && !loaded[page]) { loaded[page] = true; loaders[page](); }
}
$$("#nav button").forEach((b) => b.onclick = () => show(b.dataset.page));

/* ── 1. overview ─────────────────────────────────────────────────────── */

loaders.overview = async () => {
  try {
    STATS = await api("/api/stats");
  } catch (e) { $("#statCards").innerHTML = err(e.message); return; }
  Object.assign(LABEL_TH, STATS.label_th || {});

  const g = STATS.graph, s = STATS.settings;
  $("#statCards").innerHTML = [
    ["โหนดในกราฟ", num(g.total_nodes), "Neo4j"],
    ["ความสัมพันธ์", num(g.total_rels), "typed relations"],
    ["เวกเตอร์", num(STATS.vectors), "Qdrant · vanilla baseline"],
    ["คำถามประเมิน", num(STATS.questions_total), `${STATS.eval_files.length} ชุด`],
    ["Retriever", STATS.retrievers.length, STATS.retrievers.join(" · ")],
    ["hops", s.graph_hops, "GRAPH_HOPS"],
    ["top-k", s.top_k, "TOP_K"],
    ["LLM", esc(s.llm_model), "grounding + judge"],
  ].map(([k, v, note]) =>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${v}</div>
     <div class="note">${esc(note)}</div></div>`).join("");

  const svc = STATS.services;
  $("#svcStatus").innerHTML = Object.entries(svc).map(([k, v]) =>
    `<span class="pill ${String(v).startsWith("up") || v === "configured" ? "up" : "down"}">${esc(k)}</span>`
  ).join("");

  $("#labelTable").innerHTML =
    `<thead><tr><th>label</th><th>ไทย</th><th class="num">จำนวน</th></tr></thead><tbody>` +
    Object.entries(g.nodes).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td class="muted">${esc(LABEL_TH[k] || "")}</td>
       <td class="num">${num(v)}</td></tr>`).join("") + "</tbody>";

  $("#relTable").innerHTML =
    `<thead><tr><th>relation</th><th class="num">จำนวน</th></tr></thead><tbody>` +
    Object.entries(g.relationships).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td class="num">${num(v)}</td></tr>`).join("") + "</tbody>";

  $("#evalTable").innerHTML =
    `<thead><tr><th>ไฟล์</th><th class="num">ข้อ</th><th>แยกตาม hop</th></tr></thead><tbody>` +
    STATS.eval_files.map((f) =>
      `<tr><td>${esc(f.name)}</td><td class="num">${f.count}</td>
       <td class="small muted">${Object.entries(f.by_hop).map(([k, v]) => `${k}:${v}`).join(" · ")}</td></tr>`
    ).join("") + "</tbody>";
};

/* ── 2. ask / compare ────────────────────────────────────────────────── */

const RETR_TH = {
  vanilla: "ค้นเวกเตอร์บน Qdrant (ไม่มีโครงสร้างกราฟ)",
  graphrag: "entity-link → เดิน subgraph บน Neo4j",
  temporal: "GraphRAG + กรองด้วยช่วงเวลา",
  halal_ingredient: "เดินเส้นทาง ส่วนผสม → แหล่งที่มา → คำวินิจฉัย",
};

async function doAsk() {
  const q = $("#askInput").value.trim();
  if (!q) return;
  const retrievers = $$(".askRetr:checked").map((c) => c.value);
  if (!retrievers.length) { $("#askResults").innerHTML = err("เลือก retriever อย่างน้อยหนึ่งตัว"); return; }

  $("#askBtn").disabled = true;
  $("#askResults").innerHTML = `<div class="card muted">กำลังถาม <span class="spinner"></span></div>`;
  $("#linkTrace").innerHTML = "";

  try {
    const [res, trace] = await Promise.all([
      api("/api/ask", { method: "POST", body: JSON.stringify({ question: q, retrievers }) }),
      api(`/api/link?q=${encodeURIComponent(q)}`).catch(() => null),
    ]);

    if (trace && trace.seeds.length) {
      $("#linkTrace").innerHTML = `<div class="card"><h3>Entity linking — คำถามนี้ผูกกับโหนดใดบ้าง</h3>
        <div class="row tight">${trace.seeds.map((s) =>
          `<span class="pill on">${esc(s.name)} <span class="muted">[${esc(LABEL_TH[s.label] || s.label)}]
           ${fx(s.score, 2)} · ${esc(s.matched_by || "")}</span></span>`).join("")}</div></div>`;
    }

    $("#askResults").innerHTML = res.results.map((r) => {
      const m = r.meta || {}, u = r.usage || {};
      const bits = [
        ["seeds", m.seeds], ["ข้อเท็จจริง", m.triples], ["bridges", m.bridges],
        ["เส้นทาง", m.n_paths], ["as_of", m.as_of], ["k", m.k],
        ["retrieve", `${fx(m.latency_s, 2)}s`], ["ground", `${fx(r.ground_latency_s, 2)}s`],
        ["tokens", u.total_tokens], ["ctx", `${(r.context || "").length} ตัวอักษร`],
      ].filter(([, v]) => v !== undefined && v !== null && v !== "");
      return `<div class="card">
        <div class="retr-head"><span class="retr-dot ${esc(r.retriever)}"></span>
          <span class="retr-name">${esc(r.retriever)}</span>
          <span class="muted small">${esc(RETR_TH[r.retriever] || "")}</span></div>
        ${r.error ? err(r.error) : ""}
        <div class="answer-box">${esc(r.answer)}</div>
        <div class="metrics">${bits.map(([k, v]) => `<span>${esc(k)} <b>${esc(v)}</b></span>`).join("")}</div>
        <details><summary class="small muted" style="cursor:pointer">บริบทที่ดึงมาได้ (context ที่ส่งให้ LLM)</summary>
          <pre class="ctx" style="margin-top:8px">${esc(r.context) || "(ว่าง)"}</pre></details>
      </div>`;
    }).join("");
  } catch (e) {
    $("#askResults").innerHTML = err(e.message);
  } finally { $("#askBtn").disabled = false; }
}
$("#askBtn").onclick = doAsk;
$("#askInput").onkeydown = (e) => { if (e.key === "Enter") doAsk(); };
$$(".ex").forEach((p) => p.onclick = () => { $("#askInput").value = p.textContent; doAsk(); });

/* ── 3. KG explorer ──────────────────────────────────────────────────── */

loaders.kg = async () => {
  try {
    const { labels } = await api("/api/kg/labels");
    $("#kgLabel").innerHTML = `<option value="">ทุกประเภทโหนด</option>` +
      labels.map((l) => `<option value="${esc(l.label)}">${esc(l.label)} (${num(l.count)})</option>`).join("");
  } catch (e) { /* label filter is optional */ }
  kgSearch();
};

async function kgSearch() {
  const q = $("#kgQuery").value.trim(), label = $("#kgLabel").value;
  $("#kgResults").innerHTML = `<tbody><tr><td class="muted">กำลังค้น <span class="spinner"></span></td></tr></tbody>`;
  try {
    const res = await api(`/api/kg/search?q=${encodeURIComponent(q)}&label=${encodeURIComponent(label)}&limit=25`);
    $("#kgResults").innerHTML =
      `<thead><tr><th>ชื่อ</th><th>ประเภท</th><th class="num">degree</th></tr></thead><tbody>` +
      (res.nodes.length ? res.nodes.map((n) =>
        `<tr style="cursor:pointer" data-id="${esc(n.id)}"><td>${esc(n.name)}</td>
         <td class="small muted">${esc(LABEL_TH[n.labels[0]] || n.labels[0])}</td>
         <td class="num">${num(n.degree)}</td></tr>`).join("")
        : `<tr><td class="muted">ไม่พบโหนด</td></tr>`) + "</tbody>";
    $$("#kgResults tbody tr[data-id]").forEach((tr) => tr.onclick = () => kgDetail(tr.dataset.id));
    if (q) drawGraph(q);
    if (res.nodes.length) kgDetail(res.nodes[0].id);
  } catch (e) { $("#kgResults").innerHTML = `<tbody><tr><td>${err(e.message)}</td></tr></tbody>`; }
}
$("#kgBtn").onclick = kgSearch;
$("#kgQuery").onkeydown = (e) => { if (e.key === "Enter") kgSearch(); };
$("#kgLabel").onchange = kgSearch;

async function kgDetail(id) {
  $("#kgDetail").innerHTML = `<span class="muted">กำลังโหลด <span class="spinner"></span></span>`;
  try {
    const d = await api(`/api/kg/node/${encodeURIComponent(id)}`);
    const props = Object.entries(d.node.props)
      .filter(([k]) => !["text"].includes(k))
      .map(([k, v]) => `<tr><td class="muted">${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
    const nbr = d.neighbours.map((n) =>
      `<tr><td>${n.outgoing ? "→" : "←"} <span class="pill">${esc(d.rel_th[n.rel] || n.rel)}</span></td>
       <td style="cursor:pointer;text-decoration:underline" data-id="${esc(n.id)}">${esc(n.name)}</td>
       <td class="small muted">${esc(LABEL_TH[n.labels[0]] || n.labels[0])}</td></tr>`).join("");
    $("#kgDetailTitle").textContent = d.node.props.name || d.node.labels[0];
    $("#kgDetail").innerHTML =
      `<div class="row tight" style="margin-bottom:9px">${d.node.labels.map((l) =>
        `<span class="pill on">${esc(l)}</span>`).join("")}</div>
       ${d.node.props.text ? `<p class="small">${esc(d.node.props.text)}</p>` : ""}
       <div class="table-wrap"><table><tbody>${props}</tbody></table></div>
       <h3 style="margin-top:14px">ความสัมพันธ์ (${d.neighbours.length})</h3>
       <div class="table-wrap"><table><tbody>${nbr || `<tr><td class="muted">ไม่มี</td></tr>`}</tbody></table></div>`;
    $$("#kgDetail td[data-id]").forEach((td) => td.onclick = () => kgDetail(td.dataset.id));
  } catch (e) { $("#kgDetail").innerHTML = err(e.message); }
}

/* Force-directed graph, hand-rolled so the page has zero external deps. */
async function drawGraph(q) {
  const svg = $("#graphSvg");
  svg.innerHTML = `<text x="14" y="24" fill="#93a3bd">กำลังโหลดกราฟ…</text>`;
  let data;
  try { data = await api(`/api/kg/graph?q=${encodeURIComponent(q)}&hops=1&limit=40`); }
  catch (e) { svg.innerHTML = `<text x="14" y="24" fill="#f87171">${esc(e.message)}</text>`; return; }

  $("#graphHint").textContent = `— ${data.nodes.length} โหนด · ${data.edges.length} เส้น รอบคำค้น "${q}"`;
  if (!data.nodes.length) { svg.innerHTML = `<text x="14" y="24" fill="#93a3bd">ไม่พบโหนดที่ตรงกับคำค้น</text>`; return; }

  const W = svg.clientWidth || 900, H = 430;
  const nodes = data.nodes.map((n, i) => ({
    ...n, x: W / 2 + Math.cos(i * 2.4) * (60 + i * 7),
    y: H / 2 + Math.sin(i * 2.4) * (50 + i * 5), vx: 0, vy: 0,
  }));
  const idx = new Map(nodes.map((n, i) => [n.id, i]));
  const links = data.edges.filter((e) => idx.has(e.source) && idx.has(e.target))
    .map((e) => ({ ...e, s: idx.get(e.source), t: idx.get(e.target) }));

  for (let step = 0; step < 220; step++) {
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy || 1;
      const f = 2600 / d2, d = Math.sqrt(d2);
      a.vx -= f * dx / d; a.vy -= f * dy / d; b.vx += f * dx / d; b.vy += f * dy / d;
    }
    for (const l of links) {
      const a = nodes[l.s], b = nodes[l.t];
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
      const f = (d - 105) * 0.012;
      a.vx += f * dx / d; a.vy += f * dy / d; b.vx -= f * dx / d; b.vy -= f * dy / d;
    }
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.0016; n.vy += (H / 2 - n.y) * 0.0016;
      n.x += (n.vx *= 0.82); n.y += (n.vy *= 0.82);
      n.x = Math.max(60, Math.min(W - 60, n.x)); n.y = Math.max(22, Math.min(H - 22, n.y));
    }
  }

  const COLOURS = { Province: "#f59e0b", Region: "#f97316", Mosque: "#34d399",
    Restaurant: "#60a5fa", Hotel: "#a78bfa", Store: "#f472b6", Attraction: "#22d3ee",
    Product: "#fbbf24", Ingredient: "#34d399", Source: "#94a3b8", Ruling: "#f87171" };

  svg.innerHTML =
    links.map((l) => `<line class="link" x1="${nodes[l.s].x}" y1="${nodes[l.s].y}"
       x2="${nodes[l.t].x}" y2="${nodes[l.t].y}" stroke="#3a4a6b" stroke-width="1.2"><title>${esc(l.rel_th)}</title></line>`).join("") +
    nodes.map((n) => {
      const c = COLOURS[n.label] || "#64748b";
      const r = n.is_seed ? 9 : 6;
      const short = n.name.length > 20 ? n.name.slice(0, 19) + "…" : n.name;
      return `<circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${c}"
                stroke="${n.is_seed ? "#fff" : "none"}" stroke-width="1.6">
                <title>${esc(n.name)} [${esc(n.label)}]</title></circle>
              <text x="${n.x + r + 4}" y="${n.y + 3.5}">${esc(short)}</text>`;
    }).join("");
}

/* ── 4. benchmark ────────────────────────────────────────────────────── */

loaders.benchmark = () => loadResults();

$("#benchRun").onclick = async () => {
  $("#benchRun").disabled = true;
  const body = JSON.stringify({
    suite: $("#benchSuite").value, judge: $("#benchJudge").checked,
    limit: Number($("#benchLimit").value) || 0,
  });
  try {
    await api("/api/benchmark/run", { method: "POST", body });
    pollBenchmark();
  } catch (e) { $("#benchStatus").innerHTML = err(e.message); $("#benchRun").disabled = false; }
};
$("#benchReload").onclick = () => loadResults();

async function pollBenchmark() {
  const s = await api("/api/benchmark/status").catch(() => null);
  if (!s) return;
  if (s.running) {
    $("#benchStatus").innerHTML =
      `<span class="spinner"></span> กำลังรัน suite <b>${esc(s.suite)}</b> — ${fx(s.elapsed_s, 0)}s
       <span class="muted">(หน้านี้จะอัปเดตเองเมื่อเสร็จ)</span>`;
    setTimeout(pollBenchmark, 2500);
  } else {
    $("#benchRun").disabled = false;
    $("#benchStatus").innerHTML = s.error
      ? err(s.error) : `<div class="ok">รันเสร็จแล้ว</div>`;
    loadResults();
  }
}

const SUM_COLS = [
  ["suite", "suite"], ["retriever", "retriever"], ["hop_type", "hop"], ["n", "n"],
  ["f1", "F1"], ["em", "EM"], ["containment", "contain"],
  ["context_recall", "ctx recall"], ["faithfulness", "faith"],
  ["hit_at_k", "hit@k"], ["path_validity", "path"], ["wrong_era", "wrong-era"],
  ["latency_s", "latency s"], ["total_tokens", "tokens"],
];

async function loadResults() {
  let r;
  try { r = await api("/api/benchmark/results"); }
  catch (e) { $("#benchSummary").innerHTML = `<tbody><tr><td>${err(e.message)}</td></tr></tbody>`; return; }
  if (!r.available) {
    $("#benchSummary").innerHTML = `<tbody><tr><td class="muted">${esc(r.message)}</td></tr></tbody>`;
    return;
  }

  const m = r.meta || {};
  $("#benchMeta").innerHTML = `<div class="grid cols-4">` + [
    ["รันเมื่อ", m.generated_at || "–", ""], ["ใช้เวลา", `${fx(m.elapsed_s, 0)}s`, ""],
    ["คำถาม", num(m.questions), `${num(m.rows)} แถว`],
    ["LLM", `${num(m.llm?.calls)} calls`,
     `${num((m.llm?.prompt_tokens || 0) + (m.llm?.completion_tokens || 0))} tokens · ${num(m.llm?.errors)} errors`],
  ].map(([k, v, n]) => `<div class="stat"><div class="k">${esc(k)}</div>
    <div class="v" style="font-size:19px">${esc(v)}</div><div class="note">${esc(n)}</div></div>`).join("") + `</div>`;

  const rows = r.summary;
  const present = SUM_COLS.filter(([c]) => rows.some((x) => x[c] !== null && x[c] !== undefined && !Number.isNaN(x[c])));
  $("#benchSummary").innerHTML =
    `<thead><tr>${present.map(([, l]) => `<th class="${["suite","retriever","hop"].includes(l) ? "" : "num"}">${esc(l)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((row) => `<tr${row.hop_type === "ALL" ? ' style="font-weight:600"' : ""}>` +
      present.map(([c]) => {
        const v = row[c];
        const isText = ["suite", "retriever", "hop_type"].includes(c);
        if (isText) return `<td>${esc(v)}</td>`;
        if (c === "n") return `<td class="num">${num(v)}</td>`;
        if (c === "total_tokens") return `<td class="num">${num(Math.round(v || 0))}</td>`;
        return `<td class="num">${fx(v, 3)}</td>`;
      }).join("") + "</tr>").join("") + "</tbody>";

  const CAPTIONS = {
    "f1_by_hop.png": "ผลหลัก — F1 แยกตาม hop type: ช่องว่างของ GraphRAG ควรกว้างขึ้นจาก single → multi/relational",
    "retrieval_vs_answer.png": "เพดานของการดึงข้อมูล (context recall) เทียบกับ F1 ที่ทำได้จริง",
    "temporal_wrong_era.png": "B — อัตราการตอบผิดยุค (ยิ่งต่ำยิ่งดี)",
    "temporal_f1.png": "B — F1 บนคำถามที่ขึ้นกับเวลา",
    "ingredient_path_validity.png": "C — ความถูกต้องของคำตอบและความสมบูรณ์ของเส้นทางหลักฐาน",
    "cost_latency.png": "ต้นทุน — latency และจำนวน token เฉลี่ยต่อคำถาม",
  };
  $("#benchFigures").innerHTML = r.figures.map((f) => {
    const name = f.split("/").pop();
    return `<div class="figure"><img src="${esc(f)}?t=${Date.now()}" alt="${esc(name)}">
            <div class="cap">${esc(CAPTIONS[name] || name)}</div></div>`;
  }).join("");

  const d = r.detail || [];
  $("#benchDetail").innerHTML =
    `<thead><tr><th>id</th><th>hop</th><th>retriever</th><th>คำถาม</th>
      <th>คำตอบของระบบ</th><th>gold</th><th class="num">F1</th><th class="num">ctx</th></tr></thead><tbody>` +
    d.map((x) => `<tr><td class="small">${esc(x.id)}</td><td class="small">${esc(x.hop_type)}</td>
      <td class="small">${esc(x.retriever)}</td>
      <td class="small">${esc(String(x.question || "").slice(0, 70))}</td>
      <td class="small">${esc(String(x.answer || "").slice(0, 90))}</td>
      <td class="small muted">${esc(String(x.gold || "").slice(0, 40))}</td>
      <td class="num">${fx(x.f1, 2)}</td><td class="num">${fx(x.context_recall, 0)}</td></tr>`).join("") +
    "</tbody>";
}

/* ── 5. eval set ─────────────────────────────────────────────────────── */

let evalItems = [];

loaders.eval = async () => {
  try {
    const { files } = await api("/api/eval/files");
    $("#evalFile").innerHTML = files.map((f) =>
      `<option value="${esc(f.name)}">${esc(f.name)} — ${f.count} ข้อ</option>`).join("");
    loadEval();
  } catch (e) { $("#evalMsg").innerHTML = err(e.message); }
};

async function loadEval() {
  const name = $("#evalFile").value;
  if (!name) return;
  try {
    const d = await api(`/api/eval/${encodeURIComponent(name)}`);
    evalItems = d.items;
    $("#evalEditor").value = d.items.map((i) => JSON.stringify(i)).join("\n");
    $("#evalMsg").innerHTML = d.problems.length
      ? err(`พบปัญหา ${d.problems.length} จุด: ${d.problems.slice(0, 5).join(" | ")}`)
      : `<div class="ok">โครงสร้างถูกต้อง — ${d.count} ข้อ</div>`;
    renderEval();
  } catch (e) { $("#evalMsg").innerHTML = err(e.message); }
}
$("#evalFile").onchange = loadEval;
$("#evalHop").onchange = () => renderEval();

function renderEval() {
  const hop = $("#evalHop").value;
  const items = hop ? evalItems.filter((i) => i.hop_type === hop) : evalItems;
  $("#evalTitle").textContent = `คำถาม (${items.length} จาก ${evalItems.length})`;
  $("#evalItems").innerHTML =
    `<thead><tr><th>id</th><th>hop</th><th>คำถาม</th><th>คำตอบที่ถูก</th>
      <th>aliases</th><th>เพิ่มเติม</th></tr></thead><tbody>` +
    items.map((i) => {
      const extra = [
        i.as_of ? `as_of ${i.as_of}` : "",
        i.gold_path ? `path: ${i.gold_path.join(" → ")}` : "",
        i.gold_nodes ? `nodes: ${i.gold_nodes.join(", ")}` : "",
        i.stale_answer ? `stale: ${i.stale_answer}` : "",
      ].filter(Boolean).join(" · ");
      return `<tr><td class="small">${esc(i.id)}</td>
        <td><span class="pill">${esc(i.hop_type)}</span></td>
        <td class="small">${esc(i.question)}</td>
        <td class="small">${esc(i.answer)}</td>
        <td class="small muted">${esc((i.aliases || []).join(", "))}</td>
        <td class="small muted">${esc(extra)}</td></tr>`;
    }).join("") + "</tbody>";
}

$("#evalEditToggle").onclick = () => {
  const card = $("#evalEditorCard");
  const on = card.style.display === "none";
  card.style.display = on ? "block" : "none";
  $("#evalSave").style.display = on ? "inline-block" : "none";
  $("#evalEditToggle").textContent = on ? "ปิดตัวแก้ไข" : "แก้ไขเป็น JSONL";
};

$("#evalSave").onclick = async () => {
  const lines = $("#evalEditor").value.split("\n").map((l) => l.trim()).filter(Boolean);
  let items;
  try { items = lines.map((l, i) => { try { return JSON.parse(l); }
    catch { throw new Error(`บรรทัด ${i + 1} ไม่ใช่ JSON ที่ถูกต้อง`); } }); }
  catch (e) { $("#evalMsg").innerHTML = err(e.message); return; }
  try {
    const r = await api(`/api/eval/${encodeURIComponent($("#evalFile").value)}`,
      { method: "PUT", body: JSON.stringify({ items }) });
    $("#evalMsg").innerHTML = `<div class="ok">บันทึกแล้ว ${r.saved} ข้อ</div>`;
    loadEval();
  } catch (e) { $("#evalMsg").innerHTML = err(e.message); }
};

/* ── 6. temporal ─────────────────────────────────────────────────────── */

loaders.temporal = () => { loadTimeline(); loadCoverage(); };

const syncYear = (v) => { $("#tempYear").value = v; $("#tempSlider").value = v; loadTimeline(); };
$("#tempSlider").oninput = (e) => syncYear(e.target.value);
$("#tempYear").oninput = (e) => syncYear(e.target.value);

$("#tempBtn").onclick = async () => {
  const question = $("#tempInput").value.trim();
  if (!question) return;
  $("#tempBtn").disabled = true;
  $("#tempResults").innerHTML = `<div class="card muted">กำลังถาม <span class="spinner"></span></div>`;
  try {
    const r = await api("/api/temporal/ask", { method: "POST", body: JSON.stringify({
      question, as_of: Number($("#tempYear").value), compare_baseline: true }) });
    $("#tempResults").innerHTML = r.results.map((x) => {
      const m = x.meta || {};
      return `<div class="card">
        <div class="retr-head"><span class="retr-dot ${esc(x.retriever)}"></span>
          <span class="retr-name">${esc(x.retriever)}</span>
          <span class="muted small">${x.retriever === "temporal"
            ? `กรองเฉพาะข้อเท็จจริงที่มีผล ณ พ.ศ. ${esc(r.as_of)}` : "ไม่รู้เรื่องเวลา"}</span></div>
        <div class="answer-box">${esc(x.answer)}</div>
        <div class="metrics">
          <span>as_of <b>${esc(m.as_of ?? "–")}</b></span>
          <span>กรองเวลา <b>${m.time_filtered ? "ใช่" : "ไม่"}</b></span>
          <span>หมดอายุแล้ว <b>${esc(m.expired ?? "–")}</b></span>
          <span>บังคับใช้อยู่ <b>${esc(m.in_force ?? "–")}</b></span>
          <span>ข้อเท็จจริง <b>${esc(m.triples ?? "–")}</b></span></div>
        <details><summary class="small muted" style="cursor:pointer">บริบทที่ดึงมาได้</summary>
          <pre class="ctx" style="margin-top:8px">${esc(x.context) || "(ว่าง)"}</pre></details>
      </div>`;
    }).join("");
  } catch (e) { $("#tempResults").innerHTML = err(e.message); }
  finally { $("#tempBtn").disabled = false; }
};

async function loadTimeline() {
  try {
    const r = await api(`/api/temporal/timeline?as_of=${Number($("#tempYear").value)}`);
    $("#tempTimeline").innerHTML =
      `<thead><tr><th>ชื่อ</th><th class="num">ตั้งแต่</th><th>ผู้ออก</th></tr></thead><tbody>` +
      (r.regulations.length ? r.regulations.map((x) =>
        `<tr><td class="small">${esc(x.name)}</td><td class="num">${esc(x.valid_from)}</td>
         <td class="small muted">${esc(x.issuer || "–")}</td></tr>`).join("")
        : `<tr><td class="muted">ยังไม่ได้สร้าง temporal layer — รัน scripts.build_temporal_kg</td></tr>`) +
      "</tbody>";
  } catch (e) { $("#tempTimeline").innerHTML = `<tbody><tr><td>${err(e.message)}</td></tr></tbody>`; }
}

async function loadCoverage() {
  try {
    const c = await api("/api/temporal/coverage");
    const years = c.cert_expiry_years || [];
    const max = Math.max(...years.map((y) => y.n), 1);
    $("#tempCoverage").innerHTML =
      `<p class="small">ความสัมพันธ์ที่มีช่วงเวลากำกับ:</p>
       <div class="table-wrap"><table><thead><tr><th>relation</th><th class="num">มีเวลา</th>
         <th class="num">ทั้งหมด</th></tr></thead><tbody>` +
      c.by_relation.map((r) => `<tr><td>${esc(r.rel)}</td><td class="num">${num(r.timed)}</td>
        <td class="num">${num(r.total)}</td></tr>`).join("") + `</tbody></table></div>
       <p class="small" style="margin-top:12px">ปีที่ใบรับรองฮาลาลหมดอายุ (พ.ศ.):</p>` +
      years.map((y) => `<div class="row tight small" style="margin-bottom:3px">
        <span style="width:46px">${esc(y.year)}</span>
        <span style="height:11px;width:${(y.n / max) * 190}px;background:var(--temporal);
          border-radius:3px;display:inline-block"></span>
        <span class="muted">${num(y.n)}</span></div>`).join("");
  } catch (e) { $("#tempCoverage").innerHTML = `<span class="muted small">ยังไม่ได้สร้าง temporal layer — รัน <code>python -m scripts.build_temporal_kg</code></span>`; }
}

/* ── 7. halal ingredient ─────────────────────────────────────────────── */

loaders.ingredient = () => loadIngredients();

const RULING_CLASS = { halal: "halal", haram: "haram", mashbooh: "mashbooh" };
const RULING_TH = { halal: "ฮาลาล", haram: "ไม่ฮาลาล (หะรอม)", mashbooh: "คลุมเครือ (มัชบูฮ์)" };

async function doExplain() {
  const query = $("#ingInput").value.trim();
  if (!query) return;
  $("#ingBtn").disabled = true;
  $("#ingAnswer").innerHTML = `<div class="card muted">กำลังวิเคราะห์ <span class="spinner"></span></div>`;
  $("#ingPaths").innerHTML = "";
  try {
    const r = await api("/api/ingredient/explain", { method: "POST",
      body: JSON.stringify({ query, ground: true }) });

    $("#ingAnswer").innerHTML = `<div class="card">
      <div class="retr-head"><span class="retr-dot halal_ingredient"></span>
        <span class="retr-name">คำตอบ</span>
        <span class="muted small">อ้างอิงจากเส้นทางด้านล่างเท่านั้น</span></div>
      <div class="answer-box">${esc(r.answer || "(ไม่มีคำตอบ)")}</div>
      <div class="metrics"><span>เส้นทาง <b>${esc(r.meta?.n_paths ?? 0)}</b></span>
        <span>เส้นทางสมบูรณ์ <b>${esc(r.meta?.complete_paths ?? 0)}</b></span>
        <span>tokens <b>${esc(r.usage?.total_tokens ?? "–")}</b></span></div>
      <details><summary class="small muted" style="cursor:pointer">บริบทที่ดึงมาได้</summary>
        <pre class="ctx" style="margin-top:8px">${esc(r.context) || "(ว่าง)"}</pre></details></div>`;

    const byIng = {};
    (r.paths || []).forEach((p) => (byIng[p.ingredient] ||= []).push(p));

    $("#ingPaths").innerHTML = Object.entries(byIng).map(([ing, paths]) => {
      const conflict = new Set(paths.map((p) => p.ruling)).size > 1;
      return `<div class="card ${conflict ? "conflict" : ""}">
        <h3>${esc(ing)} ${conflict
          ? `<span class="pill mashbooh">⚠ สถานะขึ้นกับแหล่งที่มา</span>` : ""}</h3>
        ${paths.map((p) => `<div class="path">
            <span class="node">${esc(p.ingredient)}</span><span class="arrow">→</span>
            <span class="node">${esc(p.source)}</span><span class="arrow">→</span>
            <span class="node ruling ${esc(RULING_CLASS[p.ruling])}">${esc(p.ruling_th)}</span>
            <span class="basis">${esc(p.basis)}${p.note ? " · " + esc(p.note) : ""}</span>
          </div>`).join("")}
      </div>`;
    }).join("") + ((r.products || []).length ? `<div class="card">
        <h3>สินค้าที่ได้รับรองฮาลาลจริงซึ่งมีส่วนผสมนี้</h3>
        <div class="table-wrap"><table><thead><tr><th>สินค้า</th><th>รหัสฮาลาล</th>
          <th>คำที่จับได้</th></tr></thead><tbody>` +
        r.products.map((p) => `<tr><td class="small">${esc(p.product)}</td>
          <td class="small mono">${esc(p.halal_code)}</td>
          <td class="small muted">${esc(p.matched_term)}</td></tr>`).join("") +
        `</tbody></table></div></div>` : "");
  } catch (e) { $("#ingAnswer").innerHTML = err(e.message); }
  finally { $("#ingBtn").disabled = false; }
}
$("#ingBtn").onclick = doExplain;
$("#ingInput").onkeydown = (e) => { if (e.key === "Enter") doExplain(); };
$$(".ing-ex").forEach((p) => p.onclick = () => { $("#ingInput").value = p.textContent; doExplain(); });

async function loadIngredients() {
  try {
    const r = await api("/api/ingredient/list");
    $("#ingCount").textContent = `— ${r.count} ข้อวินิจฉัย`;
    $("#ingTable").innerHTML =
      `<thead><tr><th>ส่วนผสม</th><th>E-number</th><th>แหล่งที่มา</th><th>คำวินิจฉัย</th>
        <th>เหตุผล</th><th class="num">สินค้าจริง</th></tr></thead><tbody>` +
      r.rulings.map((x) => `<tr>
        <td class="small">${esc(x.ingredient)}<div class="muted" style="font-size:11px">${esc(x.ingredient_en || "")}</div></td>
        <td class="small mono">${esc(x.e_number || "–")}</td>
        <td class="small">${esc(x.source)}</td>
        <td><span class="pill ${esc(RULING_CLASS[x.ruling])}">${esc(RULING_TH[x.ruling])}</span></td>
        <td class="small muted">${esc(x.basis)}</td>
        <td class="num">${num(x.products)}</td></tr>`).join("") + "</tbody>";
  } catch (e) {
    $("#ingTable").innerHTML = `<tbody><tr><td class="muted">ยังไม่ได้สร้าง ingredient layer — รัน <code>python -m scripts.build_ingredient_kg</code></td></tr></tbody>`;
  }
}

/* ── boot ────────────────────────────────────────────────────────────── */

const start = (location.hash || "#overview").slice(1);
show(["overview", "ask", "kg", "benchmark", "eval", "temporal", "ingredient"]
  .includes(start) ? start : "overview");
if (!loaded.overview) { loaded.overview = true; loaders.overview(); }
