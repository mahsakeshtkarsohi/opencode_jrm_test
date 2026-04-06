"""
Gold evaluation dataset for the JRM Media Advisor.

Contains seed questions representative of the four routing categories:
  - KB_ONLY     : definition / terminology / methodology questions
  - DATA_ONLY   : campaign performance / data queries
  - HYBRID      : requires both KB explanation and Genie data
  - OUT_OF_SCOPE: topics explicitly not supported

Each record follows the MLflow 3 GenAI evaluation format::

    {
        "inputs": {"question": "..."},
        "expectations": {
            "expected_intent":  "kb_only" | "data_only" | "hybrid" | "out_of_scope",
            "expected_facts":   [...],    # optional, for Correctness scorer
        }
    }

Usage::

    from jrm_advisor.evaluation.dataset import GOLD_DATASET
    results = mlflow.genai.evaluate(
        data=GOLD_DATASET,
        predict_fn=predict_fn,
        scorers=[...],
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Seed gold dataset
# Each entry mirrors a real stakeholder question category from AGENTS.md.
# ---------------------------------------------------------------------------

GOLD_DATASET: list[dict] = [
    # ------------------------------------------------------------------
    # KB_ONLY — definitions, methodology, KPI explanations
    # ------------------------------------------------------------------
    {
        "inputs": {
            "question": "What is the baseline methodology used in JRM campaigns?"
        },
        "expectations": {
            "expected_intent": "kb_only",
            "expected_facts": [
                "baseline",
                "pre-campaign",
                "control",
            ],
        },
    },
    {
        "inputs": {"question": "How is incremental sales calculated?"},
        "expectations": {
            "expected_intent": "kb_only",
            "expected_facts": [
                "incremental",
                "actual",
                "baseline",
            ],
        },
    },
    {
        "inputs": {"question": "Define OTS in the context of in-store media."},
        "expectations": {
            "expected_intent": "kb_only",
            "expected_facts": [
                "on-shelf",
                "availability",
                "in-store",
            ],
        },
    },
    {
        "inputs": {"question": "What does uplift mean in retail media?"},
        "expectations": {
            "expected_intent": "kb_only",
            "expected_facts": [
                "uplift",
                "incremental",
                "sales",
            ],
        },
    },
    {
        "inputs": {
            "question": "Explain how the KPI measurement works for in-store campaigns."
        },
        "expectations": {
            "expected_intent": "kb_only",
            "expected_facts": [
                "kpi",
                "measurement",
                "in-store",
            ],
        },
    },
    # ------------------------------------------------------------------
    # DATA_ONLY — campaign performance queries
    # ------------------------------------------------------------------
    {
        "inputs": {
            "question": "Show me the weekly sales results for the Heineken campaign."
        },
        "expectations": {
            "expected_intent": "data_only",
        },
    },
    {
        "inputs": {
            "question": "What was the sales uplift percentage during the Coca-Cola campaign?"
        },
        "expectations": {
            "expected_intent": "data_only",
        },
    },
    {
        "inputs": {
            "question": "Give me the top performing weeks for the Nivea in-store campaign."
        },
        "expectations": {
            "expected_intent": "data_only",
        },
    },
    {
        "inputs": {"question": "Compare actual sales vs baseline for Q1 2024."},
        "expectations": {
            "expected_intent": "data_only",
        },
    },
    # ------------------------------------------------------------------
    # HYBRID — requires definition + performance data
    # ------------------------------------------------------------------
    {
        "inputs": {
            "question": (
                "What is sales uplift and how did the Heineken campaign perform "
                "on that metric?"
            )
        },
        "expectations": {
            "expected_intent": "hybrid",
            "expected_facts": [
                "uplift",
                "incremental",
            ],
        },
    },
    {
        "inputs": {
            "question": (
                "Explain the baseline methodology and show me the weekly results "
                "for the Dove campaign."
            )
        },
        "expectations": {
            "expected_intent": "hybrid",
            "expected_facts": [
                "baseline",
            ],
        },
    },
    # ------------------------------------------------------------------
    # OUT_OF_SCOPE — explicit refusal cases
    # ------------------------------------------------------------------
    {
        "inputs": {"question": "Can you give me a store-level ROI analysis?"},
        "expectations": {
            "expected_intent": "out_of_scope",
            "expected_facts": [
                "not yet supported",
            ],
        },
    },
    {
        "inputs": {"question": "I need a ROPO analysis for the campaign."},
        "expectations": {
            "expected_intent": "out_of_scope",
            "expected_facts": [
                "not yet supported",
                "not currently supported",
            ],
        },
    },
    {
        "inputs": {"question": "What is the digital campaign performance?"},
        "expectations": {
            "expected_intent": "out_of_scope",
            "expected_facts": [
                "in-store",
                "out of scope",
            ],
        },
    },
    {
        "inputs": {"question": "Can you forecast results for our pre-launch campaign?"},
        "expectations": {
            "expected_intent": "out_of_scope",
            "expected_facts": [
                "not yet supported",
            ],
        },
    },
]
