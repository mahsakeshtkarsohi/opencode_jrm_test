"""
JRM Media Advisor — Streamlit Application

Internal tool for Jumbo Retail Media media advisors and PMs.

Features:
  - Chat interface routed through SupervisorAgent
  - Clarification flow when campaign name is ambiguous
  - Plotly chart rendering when VisualizationSpec is returned
  - Thumbs-up / thumbs-down feedback writing to Delta
  - Session history (in-memory, cleared on page reload)
  - Mock backend toggle via USE_MOCK_BACKEND=true

Authentication:
  On Databricks the app runs as a service principal (app auth). Credentials
  are auto-injected by the platform — no token handling in this file.

Layout:
  ┌───────────────────────────────────────────┐
  │  JRM Media Advisor          [Clear chat]  │
  │  ─────────────────────────────────────    │
  │  [chat history]                           │
  │  ─────────────────────────────────────    │
  │  [chart if applicable]                    │
  │  ─────────────────────────────────────    │
  │  [feedback bar]                           │
  │  ─────────────────────────────────────    │
  │  [question input]                         │
  └───────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import streamlit as st

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="JRM Media Advisor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from jrm_advisor.app.backend import AppResponse, get_backend  # noqa: E402
from jrm_advisor.app.charts import build_chart  # noqa: E402
from jrm_advisor.app.feedback import submit_feedback  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
_APP_TITLE = "JRM Media Advisor"
_PLACEHOLDER = (
    "Ask about campaign methodology, KPIs, or performance data…\n"
    "e.g. 'What is sales uplift?' or 'Show weekly sales for the Heineken campaign'"
)
_USE_MOCK = os.getenv("USE_MOCK_BACKEND", "false").lower() == "true"

# ── Session state initialisation ──────────────────────────────────────────────


def _init_session() -> None:
    """Initialise all session-state keys on first run."""
    defaults: dict[str, Any] = {
        "messages": [],  # list of {"role": str, "content": str, "response": AppResponse|None}
        "session_id": str(uuid.uuid4()),
        "pending_feedback_idx": None,  # index of the last assistant turn awaiting feedback
        "feedback_submitted": set(),  # set of turn indices that already got feedback
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Backend (cached for the session) ─────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_backend():
    return get_backend()


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _render_message(role: str, content: str) -> None:
    """Render a single chat bubble."""
    with st.chat_message(role):
        st.markdown(content)


def _render_chart(response: AppResponse) -> None:
    """Render a Plotly chart if the response carries a valid visualization spec."""
    if not response.visualization or not response.visualization.get(
        "should_visualize", False
    ):
        return
    if not response.genie_rows:
        return
    fig = build_chart(response.visualization, response.genie_rows)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def _render_data_table(response: AppResponse) -> None:
    """Show an expandable data table when Genie rows are available."""
    if not response.genie_rows:
        return
    # Only show table expander when there is NO chart (avoids duplication)
    has_chart = (
        response.visualization is not None
        and response.visualization.get("should_visualize", False)
        and bool(response.genie_rows)
    )
    if has_chart:
        return
    with st.expander("View data table", expanded=False):
        import pandas as pd  # noqa: PLC0415

        st.dataframe(pd.DataFrame(response.genie_rows), use_container_width=True)


def _render_feedback_bar(turn_idx: int, response: AppResponse) -> None:
    """Render the thumbs-up / thumbs-down widget for a given turn."""
    if turn_idx in st.session_state["feedback_submitted"]:
        st.caption("Thank you for your feedback.")
        return

    col_up, col_down, col_spacer = st.columns([1, 1, 8])
    with col_up:
        if st.button("👍", key=f"thumbs_up_{turn_idx}", help="This answer was helpful"):
            _do_feedback(turn_idx, response, "thumbs_up")
            st.rerun()
    with col_down:
        if st.button(
            "👎", key=f"thumbs_down_{turn_idx}", help="This answer was not helpful"
        ):
            st.session_state[f"show_comment_{turn_idx}"] = True
            st.rerun()

    # Optional comment box shown after thumbs-down
    if st.session_state.get(f"show_comment_{turn_idx}", False):
        comment = st.text_input(
            "What could be improved? (optional)",
            key=f"comment_input_{turn_idx}",
            placeholder="e.g. The answer was missing the uplift percentage breakdown.",
        )
        if st.button("Submit feedback", key=f"submit_comment_{turn_idx}"):
            _do_feedback(turn_idx, response, "thumbs_down", comment=comment)
            st.rerun()


def _do_feedback(
    turn_idx: int,
    response: AppResponse,
    rating: str,
    comment: str = "",
) -> None:
    """Write feedback and mark the turn as done."""
    # Look up the question from the message history (assistant turn idx → user turn idx - 1)
    messages = st.session_state["messages"]
    # Messages alternate: user(0), assistant(1), user(2), assistant(3) …
    # assistant at position turn_idx corresponds to user at turn_idx - 1
    question = ""
    if turn_idx > 0 and messages[turn_idx - 1]["role"] == "user":
        question = messages[turn_idx - 1]["content"]

    submit_feedback(
        question=question,
        answer_text=response.text,
        rating=rating,
        comment=comment,
        session_id=st.session_state["session_id"],
        resolved_campaign=response.resolved_campaign_name,
    )
    st.session_state["feedback_submitted"].add(turn_idx)
    # Clean up comment box state
    st.session_state.pop(f"show_comment_{turn_idx}", None)


# ── Main app ──────────────────────────────────────────────────────────────────


def main() -> None:
    _init_session()
    backend = _load_backend()

    # Header
    col_title, col_clear = st.columns([6, 1])
    with col_title:
        st.title(_APP_TITLE)
        if _USE_MOCK:
            st.caption("Running with mock backend — responses are deterministic.")
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear chat", key="clear_chat"):
            st.session_state["messages"] = []
            st.session_state["feedback_submitted"] = set()
            st.session_state["pending_feedback_idx"] = None
            st.rerun()

    st.divider()

    # ── Render existing conversation history ──────────────────────────────────
    messages: list[dict[str, Any]] = st.session_state["messages"]

    for i, msg in enumerate(messages):
        _render_message(msg["role"], msg["content"])
        if msg["role"] == "assistant" and msg.get("response") is not None:
            resp: AppResponse = msg["response"]
            _render_chart(resp)
            _render_data_table(resp)
            # Only show feedback widget on the last assistant turn
            if i == len(messages) - 1:
                _render_feedback_bar(i, resp)

    # ── Chat input ────────────────────────────────────────────────────────────
    if question := st.chat_input(_PLACEHOLDER):
        # Append user message
        st.session_state["messages"].append(
            {"role": "user", "content": question, "response": None}
        )
        _render_message("user", question)

        # Call backend with a spinner
        with st.spinner("Thinking…"):
            try:
                response = backend.ask(question)
            except Exception as exc:  # noqa: BLE001
                response = AppResponse(
                    text=(
                        "An unexpected error occurred while processing your question. "
                        f"Please try again or contact your JRM media advisor.\n\n"
                        f"_(Error: {exc})_"
                    )
                )

        # Append assistant message
        st.session_state["messages"].append(
            {"role": "assistant", "content": response.text, "response": response}
        )

        # Render the new answer
        _render_message("assistant", response.text)
        _render_chart(response)
        _render_data_table(response)

        # Feedback widget for this new turn
        turn_idx = len(st.session_state["messages"]) - 1
        _render_feedback_bar(turn_idx, response)


if __name__ == "__main__":
    main()
