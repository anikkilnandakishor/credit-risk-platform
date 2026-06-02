"""Streamlit custom styling."""

import streamlit as st

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    .platform-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 55%, #0ea5e9 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
    }
    .platform-header h1 {
        color: white !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .platform-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0.35rem 0 0 0 !important;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .risk-high {
        color: #dc2626;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 700;
        font-size: 1.1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        min-width: 17rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #f8fafc !important;
        font-size: 1rem !important;
    }

    .sidebar-nav-heading {
        color: #94a3b8 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 0.35rem 0 !important;
    }

    /* Sidebar nav: radio dot + visible page name */
    section[data-testid="stSidebar"] [data-testid="stRadio"] {
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.15rem !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.65rem !important;
        width: 100% !important;
        padding: 0.4rem 0.35rem !important;
        margin: 0 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
        background: rgba(148, 163, 184, 0.12) !important;
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: rgba(139, 92, 246, 0.18) !important;
        color: #e9d5ff !important;
        font-weight: 600 !important;
    }
    /* Keep option text visible (was clipped / hidden on narrow sidebar) */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:last-child,
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label span {
        color: inherit !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        width: auto !important;
        max-width: 100% !important;
        overflow: visible !important;
        white-space: nowrap !important;
        font-size: 0.88rem !important;
        line-height: 1.3 !important;
    }
    /* Radio circle = nav dot */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
        flex-shrink: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label input[type="radio"] {
        accent-color: #8b5cf6 !important;
        width: 0.55rem !important;
        height: 0.55rem !important;
        margin: 0 !important;
    }

    .chat-user {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="platform-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
