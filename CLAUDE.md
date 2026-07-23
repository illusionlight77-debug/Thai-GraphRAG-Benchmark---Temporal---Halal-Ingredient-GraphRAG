# CLAUDE.md — Thai GraphRAG + Benchmark (Temporal · Halal-Ingredient)

> คู่มือสำหรับ Claude / นักพัฒนา ก่อนแก้โปรเจกต์นี้
> โปรเจกต์วิจัยหนึ่งตัวใหญ่: **A = Thai GraphRAG + Benchmark** เป็นแกน แล้วมี
> **B = Temporal GraphRAG** และ **C = Halal-Ingredient GraphRAG** เป็น **extension ที่แยกใช้งานได้**
> โครงนี้เป็น **seed** — โครงสร้าง/โค้ดแกนครบ รันทดลองได้เมื่อมี KG + LLM key ให้ไปทำต่อใน Claude Code

---

## 1. โปรเจกต์นี้คืออะไร (สรุป 30 วินาที)

ตอบคำถามภาษาไทยจาก **Knowledge Graph** โดยเทียบ 2 วิธีดึงข้อมูล:
- **Vanilla RAG** — embed คำถาม → ค้น vector บน Qdrant (node text) → ให้ LLM สรุป
- **GraphRAG** (แกนของงาน) — entity-link คำถาม → เดิน subgraph k-hop บน Neo4j → เอา "ข้อเท็จจริงเชิงกราฟ" ให้ LLM

**หัวใจของงานวิจัย = benchmark ที่วัดว่า GraphRAG ชนะ vanilla ตรงไหน** โดยแยกตาม hop type
(single / multi / relational) เพื่อพิสูจน์ **multi-hop advantage** ในภาษาไทย + ปล่อยชุดประเมินไทยเป็นของใหม่

**Extensions (แยกออกไปได้):**
- **B Temporal** — ใส่มิติเวลาใน relation แล้วตอบ "ณ ปีไหน" (โฟลเดอร์ `extensions/temporal/`)
- **C Halal-Ingredient** — เดินกราฟ ส่วนผสม→ที่มา→คำวินิจฉัย ตอบแบบ **อธิบายเส้นทางเหตุผลได้** (`extensions/halal_ingredient/`)

---

## 2. ⭐ กติกาดีไซน์ที่ห้ามทำผิด (design invariants)

- **การเทียบต้องยุติธรรม** — vanilla กับ graphrag ใช้ pipeline เดียวกัน (`pipeline/answer.py`),
  LLM/embedding/prompt เดียวกัน ต่างกันแค่ **retriever** เท่านั้น (คืน `RetrievedContext` เหมือนกัน)
- **แกน (A) ต้องรันได้อิสระ** — B และ C เป็น extension ห้ามแก้ core จนแกนพัง; ต่อผ่าน `retrievers.get_retriever` + benchmark
- **embedding = local bge-m3 ผ่าน TEI** (HTTP `EMBEDDING_URL`) — ให้เหมือนโปรเจกต์ chatbot, ไม่พึ่ง embed API ภายนอก
- **LLM ผ่าน endpoint OpenAI-compatible** (`LLM_BASE_URL`) — Groq เป็น default สลับเป็น OpenThaiGPT ได้
- **ไม่มี secret hardcode** — อ่านจาก `.env` ผ่าน `config.py`
- **Neo4j = แหล่ง KG เดียว**; **Qdrant ใช้เฉพาะ baseline vanilla** (index node text) — อย่าเอา graph logic ไปไว้ใน Qdrant

---

## 3. Tech Stack

| ชั้น | เทคโนโลยี |
|------|-----------|
| Knowledge Graph | Neo4j (Cypher) |
| Vector store (baseline) | Qdrant |
| Embeddings | bge-m3 ผ่าน TEI container (HTTP, 1024d) |
| LLM grounding + judge | OpenAI-compatible (Groq `llama-3.1` default / OpenThaiGPT) |
| Data/analysis | pandas, numpy, matplotlib, seaborn |
| Deploy (infra) | Docker Compose (neo4j + qdrant + embeddings) |

---

## 4. โครงสร้างโปรเจกต์

```
thaigraphrag/
├── config.py                 Settings (.env) + paths
├── app/                        ⭐ FastAPI + static SPA (7 หน้า) + REST API + /docs
│   ├── main.py                   ทุก endpoint ยิงเข้าระบบจริง ไม่มี mock
│   └── static/                   index.html · app.js · style.css (ไม่มี CDN)
├── core/
│   ├── embeddings.py           bge-m3 ผ่าน TEI (auto-split 413, retry, wait_ready)
│   ├── llm.py                  ground_detailed() + judge_faithfulness() + Usage (นับ token จริง)
│   ├── neo4j_client.py         driver + run(cypher)
│   ├── qdrant_client.py        vector store baseline (collection kg_nodes)
│   └── entity_linking.py       ⭐ query → seed nodes (gazetteer → fulltext → n-gram, Dice)
├── kg/
│   ├── provinces.py            ⭐ 77 จังหวัด + geocode point-in-polygon (กู้ LOCATED_IN)
│   ├── schema.py               SourceSpec/EdgeSpec แบบ declarative + node_text()
│   └── build_kg.py             engine: CSV → Neo4j + Qdrant (province cascade 3 ชั้น)
├── retrievers/
│   ├── base.py                 ⭐ Retriever ABC + RetrievedContext (seam ของการเทียบ)
│   ├── vanilla.py              baseline: Qdrant top-k
│   ├── graphrag.py             ⭐ entity-link → seed card + aggregated hop + bridge path
│   └── __init__.py             get_retriever() — extension ลงทะเบียนที่นี่ (lazy import)
├── pipeline/answer.py          retrieve → ground (เหมือนกันทุก retriever, ห้าม branch)
├── benchmark/
│   ├── datasets.py             โหลด/บันทึก/ตรวจ eval .jsonl
│   ├── metrics.py              F1(char)/EM/containment + context_recall/hit@k/path_validity
│   └── run_benchmark.py        ⭐ 3 suite (core/temporal/ingredient) → results/ + กราฟ
└── extensions/
    ├── temporal/               B: temporal_kg.py (valid_to จริงจาก expire_date) + retriever
    └── halal_ingredient/       C: ingredient_kg.py + explain_retriever.py (คืน path จริง)

data/
├── *.csv                       source CSV (ไม่ commit — ดู data/README.md)
├── thailand_provinces.json     ขอบเขต 77 จังหวัด (geocode)
├── halal/ingredient_rulings.csv    ⭐ 90 คำวินิจฉัย · 52 ส่วนผสม · 34 แหล่งที่มา (commit)
├── halal/regulation_timeline.csv   ไทม์ไลน์ระเบียบ (commit)
└── questions/*.jsonl           ชุดประเมิน — สร้างจากกราฟด้วย scripts.generate_eval

scripts/  build_kg · build_temporal_kg · build_ingredient_kg · generate_eval
          run_benchmark · bootstrap (ใช้ตอนบูต Docker)
Dockerfile · docker-compose.yml · pyproject.toml · requirements.txt · .env.example
docs/METHODOLOGY.md · docs/REFERENCES.md · FILE_DIRECTORY.md
```

---

## 5. วิธีรัน

### 🐳 คำสั่งเดียวจบ (แนะนำ)

```bash
cp .env.example .env      # ใส่ LLM_API_KEY; วาง CSV ไว้ใน ./data
docker compose up --build -d
```

ขึ้นครบทั้ง **neo4j + qdrant + embeddings(TEI) + app**
บูตครั้งแรก `app` จะรอ service พร้อม → สร้าง KG → สร้าง layer B/C → รัน benchmark หนึ่งรอบ → เสิร์ฟ

| URL | คืออะไร |
|-----|---------|
| http://localhost:8000 | Demo UI (7 หน้า) |
| http://localhost:8000/docs | Swagger |
| http://localhost:8000/health | สถานะ dependency |
| http://localhost:7474 | Neo4j Browser (neo4j / `NEO4J_PASSWORD`) |
| http://localhost:6333/dashboard | Qdrant dashboard |
| http://localhost:8080 | TEI (bge-m3) |

บูตครั้งแรกช้า: TEI โหลดโมเดล ~2GB และ KG build บน CPU ใช้เวลาเป็นชั่วโมง
รอบถัดไปตั้ง `SKIP_BOOTSTRAP=1` ใน `.env`

### รันบนเครื่อง (host) สำหรับพัฒนา

```bash
docker compose up -d neo4j qdrant embeddings
pip install -r requirements.txt

python -m scripts.build_kg --reset               # KG + vector index
python -m scripts.build_temporal_kg              # extension B
python -m scripts.build_ingredient_kg            # extension C
python -m scripts.generate_eval --suite all      # ชุดคำถาม (gold มาจากกราฟ)
python -m scripts.run_benchmark --suite all --judge
pytest                                            # 55 tests, ไม่ต้องต่อ DB
uvicorn thaigraphrag.app.main:app --port 8000
```

ผลออกที่ `results/`: `benchmark_detail.csv`, `benchmark_summary.csv`,
`benchmark_meta.json`, `figures/*.png`

---

## 6. สถานะการทำงาน (ทำเสร็จแล้ว / ข้อจำกัดที่เหลือ)

**A (แกน) — เสร็จ**
1. ชุดประเมินไทยขยายแล้ว สร้างจากกราฟจริงด้วย `scripts.generate_eval` (gold ถูกโดยโครงสร้าง)
2. entity linking: gazetteer (มีเงื่อนไข prefix กัน false positive อย่าง "กลาง") → Neo4j fulltext → n-gram fallback (cap 12 span), ให้คะแนนด้วย Dice
3. GraphRAG: seed card + **aggregated** neighbourhood (นับ+ตัวอย่าง) + bridge path — เลิกใช้ `MATCH path=(s)-[*1..k]-(m)` ที่ระเบิดบนโหนด hub
4. metric ครบ: `context_recall`, `hit_at_k`, `path_validity`, token/call จริงจาก API

**B (Temporal) — เสร็จ** `valid_to` จาก `expire_date` จริง 100% ของทะเบียน 222k แถว
(ตั้งใจปล่อย `valid_from` เป็น NULL เพราะข้อมูลไม่มีวันออกใบรับรอง) + timeline ระเบียบ 5 รายการ

**C (Halal-Ingredient) — เสร็จ** 90 คำวินิจฉัย, 22/52 ส่วนผสมมีคำวินิจฉัยต่างกันตามแหล่งที่มา,
`CONTAINS` สกัดจากชื่อสินค้าจริง

**ข้อจำกัดที่ยังอยู่** (รายละเอียดใน `docs/METHODOLOGY.md` §Known limitations)
- gold answer สืบทอดความผิดพลาดของข้อมูลต้นทาง (พิกัด OSM ผิด → จังหวัดผิด) — ตรวจย้อนได้จาก `province_source`
- `valid_from` ของใบรับรองไม่ทราบ → คำถามจำกัดอยู่ในช่วงปีที่ `valid_to` แยกแยะได้
- ตาราง ruling เป็น research artefact ไม่ใช่คำฟัตวา — ข้อที่มีทัศนะต่างกันจัดเป็น `mashbooh`
- F1 ระดับตัวอักษรให้คะแนนบางส่วนกับคำตอบยาว จึงรายงาน EM/containment คู่กันเสมอ

---

## 7. อ้างอิงงานวิจัย
ดู `docs/REFERENCES.md` (RAG vs GraphRAG, TG-RAG/ECT-QA, NodeRAG/LightRAG, OpenThaiGPT ฯลฯ)
