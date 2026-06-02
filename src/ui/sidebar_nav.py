"""Sidebar navigation with dot indicators and page names."""

from __future__ import annotations

import streamlit as st

SESSION_KEY = "current_page"


def render_navigation(page_options: list[str]) -> str:
    """
    Render vertical nav: colored dot + page name per row.

    Returns the selected page name.
    """
    if not page_options:
        raise ValueError("page_options must not be empty")

    if SESSION_KEY not in st.session_state or st.session_state[SESSION_KEY] not in page_options:
        st.session_state[SESSION_KEY] = page_options[0]

    st.markdown('<p class="sidebar-nav-heading">Navigation</p>', unsafe_allow_html=True)

    selected = st.radio(
        "Navigation",
        page_options,
        index=page_options.index(st.session_state[SESSION_KEY]),
        key="sidebar_page_radio",
        label_visibility="collapsed",
    )
    st.session_state[SESSION_KEY] = selected
    return selected
