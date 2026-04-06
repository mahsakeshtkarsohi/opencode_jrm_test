"""Knowledge Base sub-module — Databricks KA Model Serving client."""

from jrm_advisor.knowledge_base.client import (
    ANSWER_UNAVAILABLE,
    KnowledgeBaseClient,
    KnowledgeBaseEmptyResponseError,
    KnowledgeBaseError,
    KnowledgeBaseTimeoutError,
)

__all__ = [
    "KnowledgeBaseClient",
    "KnowledgeBaseError",
    "KnowledgeBaseTimeoutError",
    "KnowledgeBaseEmptyResponseError",
    "ANSWER_UNAVAILABLE",
]
