"""
JRM Media Advisor — CLI / Databricks job entrypoint.

This module is the ``main`` entry point referenced in ``databricks.yml``
(python_wheel_task.entry_point = "main").

Usage (local interactive):
    python -m jrm_advisor

Usage (Databricks job):
    Launched automatically by the platform via the python_wheel_task.

Environment variables required:
    DATABRICKS_HOST    — workspace URL
    DATABRICKS_TOKEN   — PAT or service principal OAuth token
    GENIE_SPACE_ID     — Genie Space ID for in-store campaign data
    KB_ENDPOINT_URL    — Knowledge Base Model Serving endpoint invocation URL

Copy .env.example to .env and fill in values for local use.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from jrm_advisor.supervisor.agent import SupervisorAgent, VisualizationSpec

# Load .env for local development (no-op in production Databricks jobs
# where env vars are injected by the platform).
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

_BANNER = """\
╔══════════════════════════════════════════════════════╗
║       JRM Media Advisor — Jumbo Retail Media         ║
║   In-store campaign analysis · type 'exit' to quit   ║
╚══════════════════════════════════════════════════════╝"""

_SEPARATOR = "─" * 54


def _render_response(agent_response) -> None:
    """Print a SupervisorResponse to stdout in a readable format."""
    print()
    print(_SEPARATOR)
    print(agent_response.text)

    viz = agent_response.visualization
    if viz and isinstance(viz, VisualizationSpec) and viz.should_visualize:
        print()
        print(
            f"[Chart available: {viz.chart_type} | "
            f"x={viz.x_field} | y={viz.y_field} | "
            f"format={viz.y_format} | scale={viz.scale}]"
        )
    print(_SEPARATOR)
    print()


def run_interactive() -> None:
    """Start an interactive REPL session with the JRM Media Advisor."""
    print(_BANNER)
    print()

    try:
        agent = SupervisorAgent()
    except ValueError as exc:
        logger.error("Failed to initialise JRM Media Advisor: %s", exc)
        print(
            "ERROR: Missing environment variables. "
            "Copy .env.example to .env and fill in all required values.",
            file=sys.stderr,
        )
        sys.exit(1)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            response = agent.answer(question)
            _render_response(response)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error while processing question: %s", exc)
            print(
                "\nAn unexpected error occurred. Please try again.",
                file=sys.stderr,
            )


def main() -> None:
    """Databricks python_wheel_task entry point."""
    run_interactive()


if __name__ == "__main__":
    main()
