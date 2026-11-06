"""Streamlit client for PDF indexing and grounded question answering."""

from __future__ import annotations

import html
import math
import os
from typing import Any, BinaryIO

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_TIMEOUT = httpx.Timeout(600.0, connect=15.0)
ANSWER_TIMEOUT = httpx.Timeout(180.0, connect=15.0)
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_PDF_SIZE_BYTES", str(50 * 1024 * 1024)))
MAX_UPLOAD_SIZE_MB = math.ceil(MAX_UPLOAD_SIZE_BYTES / (1024 * 1024))

st.set_page_config(
    page_title="Bayyinah AI | Arabic Document Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

    :root {
        --ink: #102a2a;
        --muted: #647777;
        --paper: #F8FAFC;
        --teal: #0F766E;
        --teal-dark: #115E59;
        --accent: #14B8A6;
        --line: #e2e9e9;
        --card: #FFFFFF;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Arabic', 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: var(--paper);
        direction: ltr;
    }
    [data-testid="stSidebar"] {
        direction: rtl;
    }
    .main .block-container {
        direction: rtl;
    }
    [data-testid="stHeader"] { background: rgba(248, 250, 252, .92); }
    [data-testid="stSidebar"] {
        background: #f1f6f5;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .8rem; }
    .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.15rem !important; line-height: 1.3 !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.08rem !important; }
    p, label, .stCaption { color: var(--muted); }
    .brand-lockup {
        padding: .6rem .25rem 1rem;
        border-bottom: 1px solid #dce7e5;
        margin-bottom: .35rem;
    }
    .brand-mark {
        color: var(--teal);
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1;
    }
    .brand-kicker {
        color: var(--muted);
        font-size: .76rem;
        margin-top: .5rem;
    }
    .welcome-panel {
        position: relative;
        overflow: hidden;
        border: 1px solid #dcebe8;
        border-radius: 22px;
        background: linear-gradient(115deg, #ffffff 12%, #f0f8f6 100%);
        padding: 1.7rem 2rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 12px 32px rgba(20, 54, 51, .055);
    }
    .welcome-panel::after {
        content: '';
        position: absolute;
        left: -2.5rem;
        bottom: -5rem;
        width: 13rem;
        height: 13rem;
        border-radius: 50%;
        border: 1px solid rgba(20, 184, 166, .13);
        box-shadow: 0 0 0 1.5rem rgba(20, 184, 166, .035), 0 0 0 3rem rgba(20, 184, 166, .025);
        pointer-events: none;
    }
    .welcome-kicker {
        color: var(--teal);
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }
    .welcome-subtitle { color: var(--muted); margin: .15rem 0 0; font-size: .95rem; }
    .welcome-tagline { color: #365b57; margin: .8rem 0 0; font-size: .95rem; }
    .section-heading {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin: 1.25rem 0 .7rem;
        color: #244440;
        font-size: 1rem;
        font-weight: 700;
    }
    .empty-state {
        text-align: center;
        border: 1px dashed #cbdcd8;
        border-radius: 20px;
        background: rgba(255, 255, 255, .72);
        padding: 2rem 1.5rem;
        margin: .7rem 0 1rem;
    }
    .empty-state-icon { font-size: 2.1rem; line-height: 1.2; margin-bottom: .55rem; }
    .empty-state-title { color: var(--teal-dark); font-size: 1.08rem; font-weight: 700; }
    .empty-state-copy { max-width: 35rem; margin: .45rem auto 0; color: var(--muted); line-height: 1.9; }
    .assistant-empty {
        margin: 1.2rem auto;
        color: var(--muted);
        text-align: center;
        font-size: .95rem;
    }
    .assistant-empty span { color: var(--teal); margin-inline-end: .35rem; }
    div.stButton > button[kind="primary"] {
        background: var(--teal);
        border: 1px solid var(--teal);
        color: white;
        border-radius: 13px;
        font-weight: 600;
        min-height: 2.75rem;
        transition: .16s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        transform: translateY(-1px);
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 13px;
        border-color: #dbe7e5;
        color: var(--teal-dark);
        min-height: 2.8rem;
        background: #ffffff;
        white-space: normal;
        transition: .16s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent);
        background: #f1fbf9;
        color: var(--teal-dark);
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b7cfca;
        border-radius: 15px;
        padding: .45rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3ebea;
        border-radius: 16px;
        padding: .75rem .9rem;
        box-shadow: 0 6px 18px rgba(20, 54, 51, .035);
        min-height: 94px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.16rem; font-weight: 700; }
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e3ebea;
        border-radius: 14px;
        overflow: hidden;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #e3ebea;
        border-radius: 17px;
        box-shadow: 0 6px 18px rgba(20, 54, 51, .035);
    }
    [data-testid="stChatMessage"] {
        width: fit-content;
        max-width: 88%;
        background: transparent;
        border: 0;
        padding: .2rem 0;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        margin-inline-end: auto;
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-inline-start: auto;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: #e6f4f1;
        border: 1px solid #d0e9e3;
        border-radius: 18px 18px 5px 18px;
        padding: .8rem 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 1px solid #e3ebea;
        border-radius: 5px 18px 18px 18px;
        padding: .95rem 1.1rem;
        box-shadow: 0 7px 20px rgba(20, 54, 51, .045);
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        border-radius: 13px;
    }
    [data-testid="stTextArea"] textarea {
        background: #ffffff;
        line-height: 1.8;
        min-height: 6rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        min-height: auto;
        padding: .65rem .75rem;
        box-shadow: none;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1rem; }
    .sidebar-status {
        display: flex;
        align-items: center;
        gap: .55rem;
        padding: .7rem .8rem;
        border: 1px solid #dfe9e7;
        border-radius: 13px;
        background: #ffffff;
        color: #31524d;
        font-size: .85rem;
    }
    .status-dot {
        width: .55rem;
        height: .55rem;
        flex: 0 0 .55rem;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 3px rgba(20, 184, 166, .12);
    }
    .status-dot-offline {
        background: #94a3a3;
        box-shadow: 0 0 0 3px rgba(100, 119, 119, .12);
    }
    .sidebar-file {
        overflow-wrap: anywhere;
        color: #284541;
        font-size: .88rem;
        font-weight: 600;
        line-height: 1.65;
    }
    @media (max-width: 900px) {
        .block-container { padding-top: 1rem; }
        .welcome-panel { padding: 1.3rem 1.4rem; }
        [data-testid="stChatMessage"] { max-width: 100%; }
    }
    @media (max-width: 600px) {
        .block-container { padding: .8rem .8rem 2rem; }
        h1 { font-size: 1.75rem !important; }
        .welcome-panel { padding: 1.1rem; border-radius: 17px; }
        [data-testid="stMetric"] { min-height: 82px; padding: .55rem; }
        [data-testid="stMetricLabel"] { font-size: .7rem; }
        [data-testid="stMetricValue"] { font-size: .95rem; }
    }
    .block-container { max-width: 1320px; padding-top: 1.25rem; }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 0;
        box-shadow: 1px 0 0 rgba(15, 23, 42, .06);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }
    .brand-lockup {
        padding: .3rem .1rem .75rem;
        border-bottom: 0;
        margin-bottom: .15rem;
    }
    .brand-mark { font-size: 1.35rem; }
    .brand-kicker { margin-top: .3rem; }
    .welcome-panel {
        border: 0;
        border-radius: 18px;
        background: #ffffff;
        padding: 1.25rem 1.6rem;
        margin-bottom: .8rem;
        box-shadow: 0 5px 22px rgba(15, 23, 42, .035);
    }
    .welcome-panel::after { display: none; }
    .welcome-kicker { margin-bottom: .1rem; }
    .welcome-panel h1 { margin: .15rem 0; font-size: 1.8rem !important; }
    .welcome-subtitle { font-size: .92rem; }
    .welcome-tagline { margin-top: .45rem; font-size: .86rem; }
    .section-heading {
        margin: 1rem 0 .45rem;
        font-size: .94rem;
    }
    [data-testid="stMetric"] {
        min-height: 70px;
        padding: .55rem .75rem;
        border: 0;
        border-radius: 12px;
        box-shadow: 0 3px 14px rgba(15, 23, 42, .035);
    }
    [data-testid="stMetricLabel"] { font-size: .74rem; }
    [data-testid="stMetricValue"] { font-size: 1rem; }
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="secondary"] {
        min-height: 2.25rem;
        border-radius: 10px;
        font-size: .88rem;
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed #cbd5e1;
        border-radius: 11px;
        padding: .25rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 0;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
    }
    [data-testid="stChatMessage"] { max-width: 84%; padding: .25rem 0; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: #e8f4f2;
        border: 0;
        border-radius: 18px 18px 5px 18px;
        padding: .75rem 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 0;
        border-radius: 5px 18px 18px 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 5px 18px rgba(15, 23, 42, .04);
    }
    [data-testid="stTextArea"] textarea {
        min-height: 5rem;
        border: 1px solid #dce5e4;
        background: #ffffff;
        line-height: 1.8;
    }
    [data-testid="stExpander"] {
        border: 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, .7);
    }
    .source-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .75rem;
        margin: .45rem 0 .2rem;
        padding: .7rem .85rem;
        border-radius: 11px;
        background: #f1f7f6;
        color: #244440;
        font-size: .84rem;
    }
    .source-name { overflow-wrap: anywhere; font-weight: 600; }
    .source-page { flex: 0 0 auto; color: var(--muted); font-size: .78rem; }
    @media (max-width: 600px) {
        .block-container { padding: .7rem .7rem 1.5rem; }
        .welcome-panel { padding: 1rem; }
        .welcome-panel h1 { font-size: 1.55rem !important; }
        [data-testid="stMetric"] { min-height: 66px; padding: .45rem; }
        [data-testid="stChatMessage"] { max-width: 100%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_error(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail or f"API request failed ({response.status_code}).")


def _upload_document(api_url: str, filename: str, content: BinaryIO) -> dict[str, Any]:
    content.seek(0)
    response = httpx.post(
        f"{api_url}/api/v1/documents/upload",
        files={"file": (filename, content, "application/pdf")},
        timeout=UPLOAD_TIMEOUT,
    )
    if not response.is_success:
        raise RuntimeError(_api_error(response))
    return response.json()


def _ask_question(api_url: str, question: str, top_k: int) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url}/api/v1/retrieval/answer",
        json={"query": question, "top_k": top_k},
        timeout=ANSWER_TIMEOUT,
    )
    if not response.is_success:
        raise RuntimeError(_api_error(response))
    return response.json()


def _search_documents(api_url: str, query: str, top_k: int) -> list[dict[str, Any]]:
    response = httpx.post(
        f"{api_url}/api/v1/retrieval/search",
        json={"query": query, "top_k": top_k},
        timeout=ANSWER_TIMEOUT,
    )
    if not response.is_success:
        raise RuntimeError(_api_error(response))

    results = response.json().get("results")
    if not isinstance(results, list):
        raise ValueError("Search response must contain a results list.")
    return results


@st.cache_data(ttl=15, show_spinner=False)
def _api_is_online(api_url: str) -> bool:
    try:
        response = httpx.get(f"{api_url}/health", timeout=3.0)
        response.raise_for_status()
    except (httpx.HTTPError, ValueError):
        return False
    return True


def _set_suggested_question(question: str) -> None:
    st.session_state.question_input = question


def _show_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("لم تُسترجع مصادر لهذا الرد.")
        return

    for source in sources:
        citation = source.get("citation_id", "[?]")
        filename = source.get("filename", "document.pdf")
        page_number = source.get("page_number", "?")
        score = source.get("score")
        st.markdown(
            '<div class="source-card" dir="rtl">'
            f'<span class="source-name">📄 {html.escape(str(filename))}</span>'
            f'<span class="source-page">📖 صفحة {html.escape(str(page_number))}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"{citation} · عرض المقطع"):
            st.write(source.get("text", ""))
            if score is not None:
                st.caption(f"درجة الصلة: {score:.3f}")
            st.caption(f"معرّف المستند: {source.get('document_id', '')}")


def _render_answer(content: str, sources: list[dict[str, Any]]) -> None:
    answer_text, separator, _source_text = content.partition("### المصادر")
    if "### الإجابة" in answer_text:
        answer_text = answer_text.split("### الإجابة", 1)[1]
    answer_text = answer_text.strip()
    st.markdown("### الإجابة")
    st.markdown(answer_text or content)
    st.markdown("### المصادر")
    _show_sources(sources)


def _show_search_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("لم تُعثر على مقاطع ذات صلة. جرّب صياغة سؤال آخر أو افهرس مستنداً.")
        return

    st.markdown(f"**المقاطع المسترجعة ({len(results)})**")
    for index, result in enumerate(results, start=1):
        source = result.get("source") or {}
        filename = source.get("filename", "document.pdf")
        page_number = source.get("page_number", result.get("page_number", "?"))
        with st.container():
            st.markdown(
                '<div class="source-card" dir="rtl">'
                f'<span class="source-name">📄 {index}. {html.escape(str(filename))}</span>'
                f'<span class="source-page">📖 صفحة {html.escape(str(page_number))}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.write(result.get("text", ""))
            score = result.get("score")
            if isinstance(score, (int, float)):
                st.caption(f"درجة الصلة: {score:.3f}")
            st.caption(f"معرّف المستند: {result.get('document_id', '')}")


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("model"):
            st.caption(f"النموذج: {message['model']}")
        if message["role"] == "assistant":
            _render_answer(message["content"], message.get("sources", []))
        else:
            st.markdown(message["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_document" not in st.session_state:
    st.session_state.indexed_document = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">⚖️ Bayyinah AI</div>'
        '<div class="brand-kicker">منصة ذكاء المستندات العربية</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("معلومات النظام", expanded=False):
        api_url = st.text_input(
            "عنوان خدمة API",
            value=DEFAULT_API_URL,
            help="عنوان خدمة FastAPI المتصلة بواجهة Bayyinah AI.",
        ).strip().rstrip("/")
    with st.expander("رفع وثيقة", expanded=False):
        uploaded_file = st.file_uploader(
            "اختر ملف PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"الحد الأقصى لحجم الملف {MAX_UPLOAD_SIZE_MB} ميغابايت.",
        )
        if st.button(
            "رفع وفهرسة الوثيقة",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        ):
            if uploaded_file is None:
                st.warning("اختر ملف PDF أولاً.")
            elif uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error(f"حجم الملف يتجاوز الحد الأقصى وهو {MAX_UPLOAD_SIZE_MB} ميغابايت.")
            else:
                with st.spinner("جارٍ فهرسة الوثيقة واستخراج النص..."):
                    try:
                        indexed = _upload_document(api_url, uploaded_file.name, uploaded_file)
                        st.session_state.indexed_document = indexed
                        st.session_state.messages = []
                        st.session_state.search_results = None
                        st.session_state.search_query = ""
                        st.success("أصبحت الوثيقة جاهزة للبحث.")
                    except httpx.TimeoutException:
                        st.error("استغرق الفهرس وقتاً أطول من المتوقع. تحقق من حالة API وحاول مجدداً.")
                    except httpx.HTTPError as exc:
                        st.error(f"تعذر الوصول إلى API: {exc}")
                    except RuntimeError as exc:
                        st.error(f"تعذر فهرسة الملف: {exc}")
                    except (ValueError, KeyError) as exc:
                        st.error(f"أعاد API استجابة غير متوقعة: {exc}")

    st.markdown('<div class="section-heading">الوثائق المفهرسة</div>', unsafe_allow_html=True)
    indexed_document = st.session_state.indexed_document
    if indexed_document:
        indexed_metadata = indexed_document.get("metadata", {})
        indexed_statistics = indexed_document.get("statistics", {})
        with st.container():
            st.markdown(f"**📄 {indexed_metadata.get('source_filename', 'وثيقة مفهرسة')}**")
            st.caption(
                f'📖 {indexed_statistics.get("page_count", 0)} صفحة'
                f'　·　✂️ {indexed_statistics.get("chunk_count", 0)} مقطع'
            )
            with st.expander("تفاصيل الوثيقة"):
                st.caption(f'المتجهات المفهرسة: {indexed_statistics.get("indexed_vectors", 0)}')
                st.code(indexed_document.get("document_id", ""), language=None)
    else:
        st.caption("ستظهر وثيقتك هنا بعد اكتمال الفهرسة.")

    api_online = _api_is_online(api_url) if api_url else False
    status_text = "الخدمة متصلة وجاهزة" if api_online else "تعذر الاتصال بالخدمة"
    status_class = "status-dot" if api_online else "status-dot status-dot-offline"
    st.markdown(
        f'<div class="section-heading">حالة النظام</div>'
        f'<div class="sidebar-status"><span class="{status_class}"></span>{status_text}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("إعدادات الاتصال", expanded=False):
        if st.button("اختبار الاتصال", use_container_width=True):
            try:
                health = httpx.get(f"{api_url}/health", timeout=5.0)
                health.raise_for_status()
                _api_is_online.clear()
                st.success("واجهة API متصلة.")
            except (httpx.HTTPError, ValueError) as exc:
                _api_is_online.clear()
                st.error(f"تعذر الاتصال بواجهة API: {exc}")

    with st.expander("إعدادات البحث"):
        top_k = st.slider("عدد المقاطع المسترجعة", min_value=1, max_value=10, value=5)
        st.caption("تظهر المصادر مع اسم الملف ورقم الصفحة.")

indexed_document = st.session_state.indexed_document
indexed_stats = indexed_document.get("statistics", {}) if indexed_document else {}
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

st.markdown(
    '<div class="welcome-panel" dir="rtl">'
    '<h1>⚖️ Bayyinah AI</h1>'
    '<p class="welcome-subtitle">منصة الذكاء الوثائقي العربي</p>'
    '<p class="welcome-tagline">رفع الوثائق • استرجاع المعرفة • إجابات مدعومة بالمصادر</p>'
    '</div>',
    unsafe_allow_html=True,
)

status_columns = st.columns(5)
status_columns[0].metric("📄 الوثائق", 1 if indexed_document else 0)
status_columns[1].metric("✂️ المقاطع", indexed_stats.get("chunk_count", 0))
status_columns[2].metric("🧠 النموذج النشط", model_name)
status_columns[3].metric("🗄️ مخزن المتجهات", "Qdrant")
status_columns[4].metric("✅ حالة النظام", "متصل" if api_online else "غير متصل")

if st.session_state.messages:
    st.markdown('<div class="section-heading">المحادثة</div>', unsafe_allow_html=True)
    for chat_message in st.session_state.messages:
        _render_message(chat_message)
elif indexed_document:
    st.markdown(
        '<div class="empty-state" dir="rtl">'
        '<div class="empty-state-icon">✧</div>'
        '<div class="empty-state-title">وثيقتك جاهزة للتحليل</div>'
        '<div class="empty-state-copy">اكتب سؤالك لبدء محادثة موثقة بالمصادر.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="assistant-empty" dir="rtl">'
        '<span>✦</span> ارفع وثيقة PDF من القائمة الجانبية، ثم اطرح سؤالك.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-heading">اسأل عن وثائقك</div>', unsafe_allow_html=True)
with st.form("question_form", clear_on_submit=True):
    question = st.text_area(
        "سؤالك",
        placeholder="اسأل عن محتوى وثيقتك، أو اطلب تلخيصاً أو استخراج معلومة...",
        height=88,
        label_visibility="collapsed",
        key="question_input",
    )
    ask = st.form_submit_button(
        "إرسال السؤال  ←",
        type="primary",
    )

if ask:
    if not question.strip():
        st.warning("اكتب سؤالك قبل الإرسال.")
    elif not api_url:
        st.error("أدخل عنوان خدمة API في الشريط الجانبي.")
    else:
        try:
            with st.spinner("جارٍ استرجاع السياق وكتابة إجابة موثقة..."):
                result = _ask_question(api_url, question.strip(), top_k)
            st.session_state.messages.append({"role": "user", "content": question.strip()})
            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "model": result.get("model"),
            }
            st.session_state.messages.append(assistant_message)
            st.rerun()
        except httpx.TimeoutException:
            st.error("انتهت مهلة الإجابة. قد يكون النموذج أو خدمة البحث مشغولين؛ حاول مجدداً.")
        except httpx.HTTPError as exc:
            st.error(f"تعذر الوصول إلى API: {exc}")
        except RuntimeError as exc:
            st.error(f"تعذر توليد الإجابة: {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"أعاد API استجابة غير متوقعة: {exc}")

if not st.session_state.messages:
    st.caption("جرّب أحد الاقتراحات")
    suggested_questions = ["لخص الوثيقة", "ما الهدف من الوثيقة؟", "استخرج أهم النقاط"]
    suggestion_columns = st.columns(3)
    for index, suggested_question in enumerate(suggested_questions):
        with suggestion_columns[index]:
            st.button(
                suggested_question,
                key=f"suggested_question_{index}",
                use_container_width=True,
                on_click=_set_suggested_question,
                args=(suggested_question,),
            )

with st.expander("بحث متقدم في المقاطع المسترجعة"):
    with st.form("search_form"):
        search_query = st.text_input(
            "عبارة البحث",
            placeholder="ابحث عن موضوع أو حقيقة في الوثائق المفهرسة",
        )
        search = st.form_submit_button("بحث في الوثائق", use_container_width=True)

    if search:
        if not search_query.strip():
            st.warning("اكتب عبارة للبحث.")
        elif not api_url:
            st.error("أدخل عنوان خدمة API في الشريط الجانبي.")
        else:
            try:
                with st.spinner("جارٍ استرجاع المقاطع ذات الصلة..."):
                    results = _search_documents(api_url, search_query.strip(), top_k)
                st.session_state.search_query = search_query.strip()
                st.session_state.search_results = results
            except httpx.TimeoutException:
                st.error("انتهت مهلة البحث. تحقق من حالة API وحاول مجدداً.")
            except httpx.HTTPError as exc:
                st.error(f"تعذر الوصول إلى API: {exc}")
            except RuntimeError as exc:
                st.error(f"تعذر البحث في المستندات: {exc}")
            except (ValueError, AttributeError, TypeError) as exc:
                st.error(f"أعاد API استجابة بحث غير متوقعة: {exc}")

    if st.session_state.search_results is not None:
        st.caption(f"نتائج البحث عن: {st.session_state.search_query}")
        _show_search_results(st.session_state.search_results)
