"""AI chatbot for natural language analytics (local fallback + optional LLM)."""

import streamlit as st

from src.talk_to_data import SecurityError, ask
from src.talk_to_data.llm_client import has_llm_configured
from src.utils.config import DB_PATH


def _init_chat_history():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your credit risk data assistant. "
                    "I answer common questions instantly from the local database. "
                    "For other questions, I'll use AI when an API key is configured."
                ),
            }
        ]


def _safe_ask(question: str):
    """Never raise to Streamlit — always return an assistant message dict."""
    try:
        result = ask(question)
        return {
            "role": "assistant",
            "content": result.answer,
            "sql": result.sql or None,
            "data": result.data if result.data is not None and not result.data.empty else None,
            "source": getattr(result, "source", "unknown"),
        }
    except SecurityError as e:
        return {
            "role": "assistant",
            "content": f"I couldn't process that request for security reasons: {e}",
        }
    except FileNotFoundError as e:
        return {
            "role": "assistant",
            "content": f"Database not ready: {e}. Run `python -m src.data.database --load`.",
        }
    except RuntimeError as e:
        return {
            "role": "assistant",
            "content": str(e),
        }
    except Exception as e:
        return {
            "role": "assistant",
            "content": (
                f"Something unexpected happened, but the app is still running. "
                f"Details: {e}"
            ),
        }


def render() -> None:
    st.subheader("Data intelligence chatbot")

    if not DB_PATH.exists():
        st.warning(
            f"Database not found at `{DB_PATH}`. "
            "Run: `python -m src.data.database --load`"
        )
        return

    if has_llm_configured():
        st.caption("Mode: **Local knowledge base** + AI for open-ended questions")
    else:
        st.caption(
            "Mode: **Local knowledge base only** (add API keys in `.env` for AI questions)"
        )

    _init_chat_history()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"via {msg['source']}")
            if msg.get("sql"):
                with st.expander("View SQL"):
                    st.code(msg["sql"], language="sql")
            if msg.get("data") is not None:
                with st.expander("View data"):
                    st.dataframe(msg["data"], use_container_width=True)

    examples = [
        "How many high-risk customers exist?",
        "What is the average income of defaulters?",
        "Which age group has the highest default rate?",
        "Which loan type is riskiest?",
        "Give me a portfolio risk summary.",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

    question = st.chat_input("Ask a question about your credit portfolio…")
    if st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.spinner("Analyzing…"):
            assistant_msg = _safe_ask(question)
        st.session_state.chat_messages.append(assistant_msg)
        st.rerun()
