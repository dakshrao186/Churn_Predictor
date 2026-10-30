"""Shared visual building blocks: page headers, metric cards, status chips, CSS.

Presentation only — no analytical computation. All pages compose these
helpers so styling stays consistent across the app.
"""

from __future__ import annotations

import streamlit as st

from src import config

# Small, maintainable block of scoped CSS. Selectors use Streamlit's
# documented stable data-testid hooks plus custom class names we control.
_APP_CSS = f"""
<style>
    /* Page canvas and typography */
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
    html, body, .stApp {{ background: {config.COLOR_BG}; }}
    h1, h2, h3 {{ color: {config.COLOR_TEXT}; letter-spacing: -0.01em; }}
    p, li {{ color: {config.COLOR_TEXT}; }}

    /* Metric cards */
    .cc-metric-card {{
        border: 1px solid {config.COLOR_BORDER};
        border-radius: 10px;
        padding: 0.85rem 1rem;
        background: {config.COLOR_BG};
        height: 100%;
    }}
    .cc-metric-card .cc-metric-label {{
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: {config.COLOR_TEXT_MUTED}; margin-bottom: 0.25rem; font-weight: 600;
    }}
    .cc-metric-card .cc-metric-value {{
        font-size: 1.55rem; font-weight: 700; color: {config.COLOR_TEXT}; line-height: 1.1;
    }}
    .cc-metric-card .cc-metric-hint {{ font-size: 0.75rem; color: {config.COLOR_TEXT_MUTED}; margin-top: 0.25rem; }}

    /* Dark headline surface */
    .cc-metric-card.cc-dark {{
        background: {config.COLOR_DARK_SURFACE}; border-color: {config.COLOR_DARK_SURFACE};
    }}
    .cc-metric-card.cc-dark .cc-metric-value {{ color: #FFFFFF; }}
    .cc-metric-card.cc-dark .cc-metric-label {{ color: #94A3B8; }}
    .cc-metric-card.cc-dark .cc-metric-hint {{ color: #94A3B8; }}

    /* Status chips */
    .cc-chip {{
        display: inline-block; font-size: 0.72rem; font-weight: 600;
        padding: 0.14rem 0.6rem; border-radius: 999px; margin-right: 0.4rem;
        border: 1px solid {config.COLOR_BORDER}; color: {config.COLOR_TEXT_MUTED};
        background: {config.COLOR_BG_SOFT};
    }}
    .cc-chip.cc-chip-blue {{ color: {config.COLOR_ACCENT_BLUE}; border-color: #BFDBFE; background: #EFF6FF; }}
    .cc-chip.cc-chip-green {{ color: {config.COLOR_ACCENT_GREEN}; border-color: #BBF7D0; background: #F0FDF4; }}
    .cc-chip.cc-chip-dark {{ color: #FFFFFF; background: {config.COLOR_DARK_SURFACE}; border-color: {config.COLOR_DARK_SURFACE}; }}

    /* Section panels */
    .cc-panel {{
        border: 1px solid {config.COLOR_BORDER}; border-radius: 12px;
        padding: 1rem 1.15rem; background: {config.COLOR_BG}; margin-bottom: 0.5rem;
    }}
    .cc-panel-title {{ font-weight: 700; color: {config.COLOR_TEXT}; margin-bottom: 0.2rem; }}
    .cc-note {{ color: {config.COLOR_TEXT_MUTED}; font-size: 0.85rem; }}
</style>
"""


def inject_css() -> None:
    """Apply the shared design system once per run."""
    st.markdown(_APP_CSS, unsafe_allow_html=True)


def page_header(title: str, description: str) -> None:
    """Consistent page pattern: one heading plus a one-line description."""
    st.markdown(f"## {title}")
    st.markdown(
        f"<p class='cc-note' style='margin-top:-0.6rem'>{description}</p>",
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    hint: str | None = None,
    dark: bool = False,
) -> str:
    """Return metric-card HTML; caller places it via st.markdown in columns."""
    cls = "cc-metric-card cc-dark" if dark else "cc-metric-card"
    hint_html = f"<div class='cc-metric-hint'>{hint}</div>" if hint else ""
    return (
        f"<div class='{cls}'>"
        f"<div class='cc-metric-label'>{label}</div>"
        f"<div class='cc-metric-value'>{value}</div>"
        f"{hint_html}"
        f"</div>"
    )


def metric_row(cards: list[dict]) -> None:
    """Render 1-5 metric cards in a responsive row of columns."""
    cols = st.columns(max(len(cards), 1), gap="small")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(metric_card(**card), unsafe_allow_html=True)


def status_chip(text: str, tone: str = "neutral") -> None:
    """Small inline status chip (dataset loaded, split label, threshold...)."""
    st.markdown(f"<span class='cc-chip cc-chip-{tone}'>{text}</span>", unsafe_allow_html=True)


def status_chip_row(chips: list[tuple[str, str]]) -> None:
    """Render several chips on one line."""
    html = "".join(
        f"<span class='cc-chip cc-chip-{tone}'>{text}</span>" for text, tone in chips
    )
    st.markdown(html, unsafe_allow_html=True)
