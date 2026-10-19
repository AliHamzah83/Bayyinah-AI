"""Streamlit client for PDF indexing and grounded question answering."""

from __future__ import annotations

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
    page_title="Bayyinah | Arabic Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

    :root {
        --ink: #172827;
        --muted: #637674;
        --paper: #f5f8f4;
        --teal: #087e78;
        --teal-dark: #075b57;
        --lime: #d5ef75;
        --coral: #e7795e;
        --line: #dbe5df;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'IBM Plex Sans Arabic', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background-color: var(--paper);
        background-image: radial-gradient(#d9e4dc 0.65px, transparent 0.65px);
        background-size: 18px 18px;
    }
    [data-testid="stHeader"] { background: rgba(245, 248, 244, 0.86); }
    [data-testid="stSidebar"] {
        background: #eaf1ec;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.15rem !important; line-height: 1.2 !important; }
    h2 { font-size: 1.35rem !important; }
    p, label, .stCaption { color: var(--muted); }
    .brand-lockup {
        padding: 0.65rem 0 1.35rem;
        border-bottom: 1px solid #cbd9d1;
        margin-bottom: 1.4rem;
    }
    .brand-mark {
        color: var(--teal-dark);
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1;
    }
    .brand-kicker {
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        margin-top: 0.45rem;
    }
    .hero-strip {
        border-top: 4px solid var(--teal);
        background: linear-gradient(105deg, rgba(255,255,255,.97), rgba(234,241,236,.94));
        padding: 1.4rem 1.6rem 1.25rem;
        margin-bottom: 1.5rem;
    }
    .hero-kicker {
        color: var(--teal-dark);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-copy { color: var(--muted); margin: 0.35rem 0 0; }
    .section-rule {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.75rem 0 0.8rem;
        color: var(--teal-dark);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .section-rule::after { content: ''; height: 1px; flex: 1; background: var(--line); }
    div.stButton > button[kind="primary"] {
        background: var(--teal);
        border: 1px solid var(--teal);
        color: white;
        border-radius: 5px;
        font-weight: 600;
        min-height: 2.7rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 5px;
        border-color: #b5c8be;
        color: var(--teal-dark);
        min-height: 2.45rem;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.75);
        border: 1px dashed #9db9aa;
        border-radius: 5px;
        padding: 0.35rem;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.78);
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.72);
        border: 1px solid var(--line);
        border-radius: 5px;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.82);
        border: 1px solid var(--line);
        border-radius: 5px;
    }
    .status-line { color: var(--muted); font-size: 0.83rem; padding-top: 0.3rem; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 2.5rem; }
        h1 { font-size: 1.75rem !important; }
        .hero-strip { padding: 1.1rem; }
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


def _show_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("لم تُسترجع مصادر لهذا الرد.")
        return

    st.markdown("**المصادر المسترجعة**")
    for source in sources:
        citation = source.get("citation_id", "[?]")
        filename = source.get("filename", "document.pdf")
        page_number = source.get("page_number", "?")
        score = source.get("score")
        label = f"{citation}  {filename} · صفحة {page_number}"
        with st.expander(label):
            st.write(source.get("text", ""))
            if score is not None:
                st.caption(f"درجة الصلة: {score:.3f}")
            st.caption(f"Document ID: {source.get('document_id', '')}")


def _show_search_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("لم تُعثر على مقاطع ذات صلة. جرّب صياغة سؤال آخر أو افهرس مستنداً.")
        return

    st.markdown(f"**المقاطع المسترجعة ({len(results)})**")
    for index, result in enumerate(results, start=1):
        source = result.get("source") or {}
        filename = source.get("filename", "document.pdf")
        page_number = source.get("page_number", result.get("page_number", "?"))
        label = f"[{index}]  {filename} · صفحة {page_number}"
        with st.expander(label):
            st.write(result.get("text", ""))
            score = result.get("score")
            if isinstance(score, (int, float)):
                st.caption(f"درجة الصلة: {score:.3f}")
            st.caption(f"Document ID: {result.get('document_id', '')}")


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("model"):
            st.caption(f"Generated with {message['model']}")
        st.markdown(message["content"])
        if message.get("sources") is not None:
            _show_sources(message["sources"])


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
        '<div class="brand-lockup"><div class="brand-mark">بيّنة</div>'
        '<div class="brand-kicker">Arabic document intelligence</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### اتصال الخدمات")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).strip().rstrip("/")
    if st.button("اختبار الاتصال", use_container_width=True):
        try:
            health = httpx.get(f"{api_url}/health", timeout=5.0)
            health.raise_for_status()
            st.success("واجهة API متصلة")
        except (httpx.HTTPError, ValueError) as exc:
            st.error(f"تعذر الاتصال بـ API: {exc}")

    st.markdown("---")
    st.markdown("#### إعدادات البحث")
    top_k = st.slider("عدد المقاطع المسترجعة", min_value=1, max_value=10, value=5)
    st.markdown(
        '<div class="status-line">مصادر موثقة حسب اسم الملف ورقم الصفحة</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hero-strip"><div class="hero-kicker">Arabic knowledge workspace</div>'
    '<h1>اسأل مستنداتك، لا تتصفحها صفحةً صفحة</h1>'
    '<p class="hero-copy">ارفع ملف PDF مفهرساً، ثم اطرح سؤالك واحصل على إجابة مرتبطة بمصادرها.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="section-rule">01 · فهرسة مستند</div>', unsafe_allow_html=True)
    st.subheader("أضف ملف PDF")
    uploaded_file = st.file_uploader(
        "اختر مستنداً عربياً أو متعدد اللغات",
        type=["pdf"],
        accept_multiple_files=False,
        help=f"الحد الأقصى لحجم الملف {MAX_UPLOAD_SIZE_MB} ميغابايت.",
    )
    if st.button(
        "رفع وفهرسة المستند",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        if uploaded_file is None:
            st.warning("اختر ملف PDF أولاً.")
        elif not api_url:
            st.error("أدخل عنوان FastAPI في الشريط الجانبي.")
        elif uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"حجم الملف يتجاوز الحد الأقصى وهو {MAX_UPLOAD_SIZE_MB} ميغابايت.")
        else:
            with st.status(
                "جارٍ إرسال الملف وفهرسته عبر FastAPI...",
                expanded=True,
            ) as indexing_status:
                st.write("يستخرج النظام النص، وينشئ المقاطع والمتجهات، ثم يحفظها في فهرس البحث.")
                try:
                    indexed = _upload_document(api_url, uploaded_file.name, uploaded_file)
                    st.session_state.indexed_document = indexed
                    st.session_state.messages = []
                    st.session_state.search_results = None
                    st.session_state.search_query = ""
                    indexing_status.update(
                        label="اكتملت الفهرسة وأصبح المستند قابلاً للبحث",
                        state="complete",
                        expanded=False,
                    )
                    st.success("اكتملت الفهرسة، وأصبح المستند قابلاً للبحث.")
                except httpx.TimeoutException:
                    indexing_status.update(label="انتهت مهلة الفهرسة", state="error")
                    st.error("استغرق الفهرس وقتاً أطول من المتوقع. تحقق من حالة API وحاول مجدداً.")
                except httpx.HTTPError as exc:
                    indexing_status.update(label="تعذر الاتصال بواجهة API", state="error")
                    st.error(f"تعذر الوصول إلى API: {exc}")
                except RuntimeError as exc:
                    indexing_status.update(label="تعذرت فهرسة الملف", state="error")
                    st.error(f"تعذر فهرسة الملف: {exc}")
                except (ValueError, KeyError) as exc:
                    indexing_status.update(label="استجابة API غير متوقعة", state="error")
                    st.error(f"أعاد API استجابة غير متوقعة: {exc}")

    indexed_document = st.session_state.indexed_document
    if indexed_document:
        stats = indexed_document.get("statistics", {})
        metadata = indexed_document.get("metadata", {})
        st.markdown("---")
        st.markdown(f"**{metadata.get('source_filename', 'مستند مفهرس')}**")
        metric_columns = st.columns(3)
        metric_columns[0].metric("الصفحات", stats.get("page_count", 0))
        metric_columns[1].metric("المقاطع", stats.get("chunk_count", 0))
        metric_columns[2].metric("المتجهات", stats.get("indexed_vectors", 0))
        st.caption(f"Document ID: {indexed_document.get('document_id', '')}")
    else:
        st.caption("تظهر إحصاءات الفهرسة هنا بعد نجاح رفع المستند.")

with right:
    st.markdown('<div class="section-rule">02 · سؤال وإجابة</div>', unsafe_allow_html=True)
    st.subheader("اسأل مكتبتك")

    st.markdown("#### معاينة المقاطع المسترجعة")
    with st.form("search_form"):
        search_query = st.text_input(
            "عبارة البحث",
            placeholder="ابحث عن موضوع أو حقيقة في المستندات المفهرسة",
        )
        search = st.form_submit_button("بحث في المستندات", use_container_width=True)

    if search:
        if not search_query.strip():
            st.warning("اكتب عبارة للبحث.")
        elif not api_url:
            st.error("أدخل عنوان FastAPI في الشريط الجانبي.")
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

    for chat_message in st.session_state.messages:
        _render_message(chat_message)

    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "سؤالك",
            placeholder="مثال: ما أبرز النتائج التي يذكرها التقرير؟",
            height=94,
            label_visibility="collapsed",
        )
        ask = st.form_submit_button("استرجاع السياق وتوليد الإجابة", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("اكتب سؤالاً قبل الإرسال.")
        elif not api_url:
            st.error("أدخل عنوان FastAPI في الشريط الجانبي.")
        else:
            try:
                with st.spinner("جارٍ استرجاع المقاطع وكتابة إجابة موثقة..."):
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
