"""
Evaluation package for the JRM Media Advisor.

Provides:
  - Gold dataset (seed questions covering KB, Genie, hybrid, out-of-scope)
  - Custom scorers aligned with the AGENTS.md scorer dimensions
  - Evaluation runner (run_eval.py) that calls mlflow.genai.evaluate()

Scorer dimensions (from AGENTS.md):
  1. correctness          — built-in Correctness scorer
  2. completeness         — Guidelines scorer
  3. business_usefulness  — Guidelines scorer
  4. clean_response       — custom rule-based scorer
  5. tool_routing_validity — custom trace-based scorer
"""

from jrm_advisor.evaluation.scorers import (
    clean_response,
    intent_routing_accuracy,
    response_not_empty,
)

__all__ = [
    "clean_response",
    "intent_routing_accuracy",
    "response_not_empty",
]
