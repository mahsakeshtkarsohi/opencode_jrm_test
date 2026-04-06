"""
MLflow GenAI evaluation runner for the JRM Media Advisor.

Runs ``mlflow.genai.evaluate()`` against the gold dataset using all
configured scorers. Results (metrics, traces, per-row feedback) are
logged to the MLflow experiment defined by ``MLFLOW_EXPERIMENT_NAME``
in your ``.env`` file (or the environment).

Usage (local — imports the agent directly, no deployed endpoint needed)::

    python -m jrm_advisor.evaluation.run_eval

Environment variables required (see .env.example):
  DATABRICKS_HOST          — Databricks workspace URL
  DATABRICKS_TOKEN         — PAT or service principal token
  GENIE_SPACE_ID           — Genie Space for campaign data queries
  KB_ENDPOINT_URL          — KA Model Serving endpoint URL (optional;
                             KB calls degrade gracefully when missing)
  MLFLOW_EXPERIMENT_NAME   — MLflow experiment path
                             (default: /Shared/jrm-advisor-eval)

Optional env vars (degrade gracefully when missing):
  DATABRICKS_SQL_WAREHOUSE_ID — enables campaign name resolution
  COMPOSER_ENDPOINT           — enables LLM answer composition

The runner uses the **local agent** pattern (imports SupervisorAgent
directly) so evaluation works without a deployed Model Serving endpoint.
This is the recommended approach for development iteration per the
MLflow evaluation skill.

Run tags
--------
Each run is tagged with:
  ``eval.phase``         — phase identifier (e.g. "phase-2-baseline")
  ``eval.agent_version`` — git commit SHA (short) at time of run
  ``eval.dataset_size``  — number of gold questions evaluated
"""

from __future__ import annotations

import logging
import os
import subprocess

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
)

from jrm_advisor.evaluation.dataset import GOLD_DATASET
from jrm_advisor.evaluation.scorers import (
    clean_response,
    intent_routing_accuracy,
    response_not_empty,
)
from jrm_advisor.supervisor.agent import SupervisorAgent, classify_intent

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/jrm-advisor-eval")

# Phase tag — bump this when comparing runs after a significant change.
_EVAL_PHASE: str = os.getenv("EVAL_PHASE", "phase-2-baseline")

# ---------------------------------------------------------------------------
# predict_fn — called by mlflow.genai.evaluate() for each dataset row.
# Receives **unpacked inputs (not a dict) per MLflow 3 GenAI contract.
# ---------------------------------------------------------------------------
_agent: SupervisorAgent | None = None


def _get_agent() -> SupervisorAgent:
    """Lazy-initialise the agent (once per process)."""
    global _agent
    if _agent is None:
        _agent = SupervisorAgent()
    return _agent


def predict_fn(question: str) -> dict:
    """Wrap SupervisorAgent.answer() for mlflow.genai.evaluate().

    The 'inputs' key in the gold dataset is ``{"question": "..."}`` so
    MLflow unpacks it as ``predict_fn(question="...")`` — matching this
    signature.

    Returns a dict with:
      ``text``   — the user-facing answer (read by all scorers)
      ``intent`` — the routing intent string (read by intent_routing_accuracy)
      ``needs_clarification`` — True when the agent asked for clarification
      ``resolved_campaign_name`` — the resolved CampaignNameAdj, or None
    """
    agent = _get_agent()
    response = agent.answer(question)
    # Classify intent independently so it is always present in outputs,
    # even when the supervisor short-circuits (e.g. out-of-scope).
    intent = classify_intent(question)
    return {
        "text": response.text,
        "intent": intent,
        "needs_clarification": response.needs_clarification,
        "resolved_campaign_name": response.resolved_campaign_name,
    }


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
SCORERS = [
    response_not_empty,
    clean_response,
    intent_routing_accuracy,
    Correctness(),
    Guidelines(
        name="business_language",
        guidelines=(
            "The response must be written in professional, plain business language. "
            "It must not contain internal system names, raw tool output, code "
            "snippets, or technical implementation details."
        ),
    ),
    Guidelines(
        name="completeness",
        guidelines=(
            "The response must fully address the request. "
            "If the requested data or knowledge is unavailable the response must "
            "clearly state that rather than returning a vague or empty answer."
        ),
    ),
    Guidelines(
        name="no_fabrication",
        guidelines=(
            "The response must not invent data, metrics, or facts that are not "
            "explicitly stated in available information. "
            "When data is missing the response must acknowledge the gap."
        ),
    ),
    Guidelines(
        name="out_of_scope_handled",
        guidelines=(
            "When the request covers a topic that is not supported "
            "(store-level ROI, ROPO, digital channels, pre-launch forecasting, "
            "WAP drill-down), the response must clearly state this limitation "
            "without fabricating an answer."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha_short() -> str:
    """Return the short git commit SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluation() -> mlflow.entities.Run:
    """Execute the evaluation and return the MLflow run.

    Connects to Databricks MLflow tracking, sets the experiment, runs
    all scorers against the gold dataset, and logs run tags for
    traceability.

    Returns:
        The completed MLflow ``Run`` object. Inspect
        ``run.data.metrics`` for aggregate scores.
    """
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(_EXPERIMENT_NAME)

    git_sha = _git_sha_short()
    run_tags = {
        "eval.phase": _EVAL_PHASE,
        "eval.agent_version": git_sha,
        "eval.dataset_size": str(len(GOLD_DATASET)),
        "eval.kb_endpoint_configured": str(bool(os.getenv("KB_ENDPOINT_URL"))),
        "eval.composer_configured": str(bool(os.getenv("COMPOSER_ENDPOINT"))),
        "eval.resolver_configured": str(bool(os.getenv("DATABRICKS_SQL_WAREHOUSE_ID"))),
    }

    logger.info(
        "Starting JRM Advisor evaluation — phase=%s dataset=%d questions git=%s",
        _EVAL_PHASE,
        len(GOLD_DATASET),
        git_sha,
    )
    logger.info("MLflow experiment: %s", _EXPERIMENT_NAME)
    logger.info(
        "Component availability: KB=%s composer=%s resolver=%s",
        run_tags["eval.kb_endpoint_configured"],
        run_tags["eval.composer_configured"],
        run_tags["eval.resolver_configured"],
    )

    with mlflow.start_run(tags=run_tags) as run:
        results = mlflow.genai.evaluate(
            data=GOLD_DATASET,
            predict_fn=predict_fn,
            scorers=SCORERS,
        )

    logger.info("Evaluation complete.")
    logger.info("Run ID  : %s", run.info.run_id)
    logger.info(
        "Run URL : %s/#mlflow/experiments/%s/runs/%s",
        os.getenv("DATABRICKS_HOST", ""),
        run.info.experiment_id,
        run.info.run_id,
    )
    logger.info("Metrics : %s", results.metrics)

    return results


if __name__ == "__main__":
    run_evaluation()
