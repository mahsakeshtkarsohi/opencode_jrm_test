# AGENTS.md

## What to do before starting work

1. Check the vault for specs and issues:
   `C:\Ai_Project\ai_vault\projects\opencode_jrm_test\`
2. If no spec exists, use `@pm-architect` to create one before writing any code.
3. Update the "Scaffolding" section below once a toolchain is in place.

## Project: AI Media Advisor — Jumbo Retail Media (JRM)

A **supervisor-style agent** for in-store retail media campaign analysis. Not a general
chatbot — it must behave like a JRM media advisor: precise terminology, business-grade
answers, honest about data gaps.

**Platform:** Databricks. Data in Unity Catalog. MLflow for experiment tracking and
GenAI evaluation. See global conventions at `C:\Ai_Project\opencode-config\AGENTS.md`.

---

## Architecture

```
User
 └─► Supervisor Agent
       ├─► Knowledge Base (Databricks)   — definitions, methodology, JRM terminology
       ├─► Genie (Databricks)            — structured campaign data queries
       ├─► Visualization Spec Function   — rule-based chart decision (not a BI engine)
       └─► [combined answer]
                └─► Application Layer    — renders text + charts, applies formatting
```

**Supervisor is the only component that speaks to the user.** It never forwards raw
tool output. Every final answer must be clean, business-facing prose.

---

## Routing Rules (critical — agents must follow these)

| User intent | Component to call |
|---|---|
| Definition, terminology, methodology, KPI explanation | Knowledge Base only |
| Campaign performance data, uplift, sales figures, time-series | Genie only |
| Requires explanation + measurement (e.g. "what is uplift and how did mine perform?") | Both |
| User asks for chart/graph/trend AND result rows are available | Visualization Spec Function |

**Never call the Visualization Spec Function if there are no result rows.**

---

## Genie — Behavior Rules

- Identify the campaign name in available data before querying.
- Default to the **during** period for in-store analysis unless the user specifies pre/post.
- Return weekly results as `year_week` or `week` depending on available structure.
- Do not invent metrics or drill-downs not present in available tables.

---

## Knowledge Base — Behavior Rules

- Return clean prose summaries. Never return raw extracted chunks, HTML fragments,
  citation markers (`[1]`, `[2]`), storage URLs, blob links, or parser artifacts.
- If methodology is partially available, state what is known and what is missing.

---

## Visualization Spec Function — Supported Charts Only

| Chart type | x-axis | y-axis |
|---|---|---|
| Line chart | `year_week` | `actual_sales` |
| Bar chart | `year_week` or `week` | `sales_uplift_percentage` |

Return `should_visualize = false` for anything outside this list.

Formatting contract returned to the UI layer:
- `y_format`: `percentage` / `currency` / `number`
- `decimals`
- `scale`: `identity` / `x100`

The **application layer** applies these rules during rendering — the function only
specifies them.

---

## Final Answer Rules

Every answer must:
- be professional, plain business language
- separate facts from interpretation
- define JRM terms when helpful
- give a recommendation only when evidence supports it (see allowed patterns below)
- clearly state when data is missing or not yet supported

Every answer must NOT contain:
- raw HTML, raw document fragments, footnotes, citation markers
- storage URLs, blob links, internal file links, parser artifacts
- raw tool traces, execution metadata, internal component names

**Preferred answer structure:**
1. Main insight
2. Supporting evidence
3. Interpretation
4. Recommendation *(omit if not supported)*

For definition/methodology questions: Definition → How it is measured → Known limitation.

---

## Recommendation Policy

Allowed: timing adjustments, budget focus, flagging weak periods, promo overlap,
execution quality issues, seasonality mismatch.

Not allowed: causal claims without evidence, long-term brand effect claims, tactics
not grounded in available data.

---

## Out-of-Scope Questions (known, do not invent answers)

The following are not yet supported. Acknowledge clearly, do not fabricate:

- Store-level ROI analysis
- Store-level OTS vs. sales uplift correlation
- WAP / store drill-down patterns
- ROPO analysis (not yet stakeholder-aligned)
- Pre-launch campaign forecasting

---

## Current In-Scope Channel

**In-store campaigns only.** Digital, out-of-home, and other channels are out of scope.

---

## Evaluation

Trace all conversations in MLflow. Evaluation assets to build:
- Gold dataset from real stakeholder questions (media advisors, PMs)
- Scorer dimensions: correctness, completeness, business usefulness, clean response,
  tool routing validity
- Human review loop: stakeholders provide expected answers, correction notes, ratings

---

## Scaffolding

### Install

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest tests/
```

### Run a single test file

```bash
pytest tests/visualization/test_spec.py -v
```

### Lint / format (requires pre-commit or ruff installed)

```bash
ruff check src/ tests/ --fix
ruff format src/ tests/
```

### Environment variables required

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL (e.g. `https://adb-<id>.azuredatabricks.net`) |
| `DATABRICKS_TOKEN` | Personal access token or service principal OAuth token |
| `GENIE_SPACE_ID` | Genie Space ID for in-store campaign data |
| `KB_ENDPOINT` | Model Serving endpoint name for the Knowledge Base |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment path (e.g. `/Shared/jrm-advisor-eval`) |

### Unity Catalog paths

_To be filled in when Unity Catalog tables are confirmed with the data team._

### MLflow

- **Experiment name:** set via `MLFLOW_EXPERIMENT_NAME` env var (default: `/Shared/jrm-advisor-eval`)
- **Run evaluation:** `python -m jrm_advisor.evaluation.run_eval`
- **Scorer dimensions:** `response_not_empty`, `clean_response`, `intent_routing_accuracy`, `Correctness`, `business_language`, `completeness`, `no_fabrication`, `out_of_scope_handled`
- **Gold dataset:** `src/jrm_advisor/evaluation/dataset.py` — 15 seed questions across KB, Genie, hybrid, and out-of-scope categories
- Model registry path: _To be added when a model is registered in Unity Catalog._

### Databricks Asset Bundle

- `databricks.yml` defines `dev` (single-node) and `prod` (serverless) targets.
- Deploy: `databricks bundle deploy --target dev`
- Run: `databricks bundle run jrm_advisor_job --target dev`
