import time
import json
import re
import io
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ── RAG / Vector Store imports ─────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS + JS  (KaTeX · Callouts)
# ══════════════════════════════════════════════════════════════
st.markdown(r"""
<!-- KaTeX -->
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="
    renderMathInElement(document.body,{
      delimiters:[
        {left:'$$',right:'$$',display:true},
        {left:'$', right:'$', display:false},
        {left:'\\[',right:'\\]',display:true},
        {left:'\\(',right:'\\)',display:false}
      ],throwOnError:false});
    new MutationObserver(function(){
      renderMathInElement(document.body,{
        delimiters:[
          {left:'$$',right:'$$',display:true},
          {left:'$', right:'$', display:false},
          {left:'\\[',right:'\\]',display:true},
          {left:'\\(',right:'\\)',display:false}
        ],throwOnError:false});
    }).observe(document.body,{childList:true,subtree:true});
  "></script>

<style>
.katex{font-size:1.15rem!important}
.katex-display{overflow-x:auto;padding:.5rem 0}
table{width:100%!important;border-collapse:collapse!important;margin:1rem 0!important;
  font-size:.92rem!important;border-radius:8px!important;overflow:hidden!important;
  box-shadow:0 1px 8px rgba(0,0,0,.3)!important}
thead tr{background:linear-gradient(135deg,#1a3a5c,#0d2137)!important;
  color:#e6edf3!important;font-weight:700!important}
th,td{padding:10px 14px!important;text-align:left!important;
  border-bottom:1px solid #21262d!important;border-right:1px solid #21262d!important}
th:last-child,td:last-child{border-right:none!important}
tbody tr:nth-child(even){background:#161b22!important}
tbody tr:hover{background:#1c2736!important;transition:background .15s}
.callout{border-radius:8px;padding:.75rem 1rem;margin:.8rem 0;
  display:flex;gap:.6rem;align-items:flex-start;font-size:.93rem}
.callout-icon{font-size:1.1rem;flex-shrink:0;margin-top:1px}
.callout.info   {background:#131f2e;border:1px solid #1f6feb;color:#58a6ff}
.callout.warning{background:#2b1f0a;border:1px solid #9e6a03;color:#e3b341}
.callout.success{background:#0d2119;border:1px solid #238636;color:#3fb950}
.callout.error  {background:#2a0e0e;border:1px solid #da3633;color:#f85149}
.callout.tip    {background:#1a1f2e;border:1px solid #6e40c9;color:#d2a8ff}
details.think-block{background:#0d1117;border:1px solid #30363d;border-radius:8px;margin:.6rem 0;padding:0}
details.think-block summary{padding:.5rem .8rem;cursor:pointer;color:#8b949e;font-size:.82rem;font-style:italic;list-style:none;user-select:none}
details.think-block summary::-webkit-details-marker{display:none}
details.think-block summary::before{content:"🧠 ";margin-right:4px}
details.think-block .think-body{padding:.6rem 1rem .8rem;color:#8b949e;font-size:.88rem;border-top:1px solid #21262d;font-style:italic;line-height:1.6;white-space:pre-wrap}
blockquote{border-left:4px solid #58a6ff!important;padding:.5rem 1rem!important;margin:.8rem 0!important;background:#161b22!important;border-radius:0 6px 6px 0!important;color:#8b949e!important;font-style:italic!important}
h1,h2,h3,h4{margin-top:1.2rem!important;margin-bottom:.5rem!important;font-weight:700!important;border-bottom:1px solid #21262d;padding-bottom:.3rem}
.stat-badge{display:inline-block;background:#161b22;border:1px solid #30363d;border-radius:20px;padding:2px 10px;font-size:.78rem;color:#8b949e;margin-right:6px}
.rag-badge{display:inline-block;background:#1a2f1a;border:1px solid #238636;border-radius:20px;padding:2px 10px;font-size:.78rem;color:#3fb950;margin-right:6px}
.wiki-badge{display:inline-block;background:#1a1f35;border:1px solid #6e40c9;border-radius:20px;padding:2px 10px;font-size:.78rem;color:#d2a8ff;margin-right:6px}
[data-testid="stChatMessage"]{border-radius:12px!important;margin-bottom:4px!important}
hr{border-color:#21262d!important;margin:1rem 0!important}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "🦙 General Chat": {
        "meta-llama/Llama-3.1-8B-Instruct": {
            "label": "Llama 3.1 8B", "tags": ["chat", "fast"],
            "context": "128k", "desc": "Meta's best open chat model."},
    },
    "🧠 Reasoning & Math": {
        "Qwen/Qwen2.5-7B-Instruct": {
            "label": "Qwen 2.5 7B", "tags": ["reasoning", "math"],
            "context": "128k", "desc": "Great at reasoning and multilingual."},
        "Qwen/Qwen2.5-72B-Instruct": {
            "label": "Qwen 2.5 72B ⭐", "tags": ["powerful", "best"],
            "context": "128k", "desc": "Near GPT-4 quality — best free model."},
    },
    "💻 Coding": {
        "Qwen/Qwen2.5-Coder-32B-Instruct": {
            "label": "Qwen Coder 32B ⭐", "tags": ["code", "best"],
            "context": "128k", "desc": "Best free coding model on HF."},
    },
    "🌍 Multilingual": {
        "Qwen/Qwen2.5-7B-Instruct": {
            "label": "Qwen 2.5 7B", "tags": ["Hindi", "100+ langs"],
            "context": "128k", "desc": "Best for Hindi/Hinglish."},
    },
}

ALL_MODELS: dict[str, dict] = {}
for _cat, _models in MODEL_REGISTRY.items():
    for _repo, _info in _models.items():
        ALL_MODELS[_repo] = {**_info, "category": _cat}

ASSISTANT_MODES = {
    "General": """You are a helpful, concise AI assistant.
Always format mathematical expressions using LaTeX:
- Inline math: $expression$  |  Block equations: $$expression$$
Format code in fenced blocks with language tag: ```python
Use markdown tables for comparisons.
Use **bold** for key terms, italics for emphasis, and blockquotes for important notes. At the end give tips💡and suggestions in bullet points.""",

    "Python Expert": """You are a senior Python developer.
Provide clean, production-quality, well-commented code.
Always use fenced code blocks: ```python
Format equations with LaTeX. Use tables for comparisons.
Highlight important notes with: > At the end give tips 💡 and suggestions in bullet points.""",

    "Data Scientist": """You are an expert in ML, Data Science, AI and GenAI.
Use precise technical explanations.
ALWAYS render mathematical formulas using LaTeX:
- Use $$...$$ for display equations  |  $...$ for inline math
Format code with proper language tags. Use markdown tables.""",

    "Interviewer": """You are a professional technical interviewer.
Ask focused questions and evaluate answers critically.
Use **bold** for question headers, numbered lists for steps.""",

    "Career Coach": """You help people with career guidance, resume tips,
interview preparation, and professional communication.
Use bullet points and structured sections.""",

    "Research Assistant": """You provide thorough, well-sourced research summaries.
Mention confidence levels where appropriate.
Use LaTeX for formulas, tables for data, headers for sections.""",

    "Explain Like I'm 5": """Explain concepts very simply using analogies
a child can understand. Use simple words and fun examples.""",

    "Socratic Tutor": """Guide users through questions so they discover
answers themselves. Use LaTeX for equations, numbered steps for reasoning.""",

    "Custom": "",
}

SUGGESTIONS = [
    "Explain transformers in simple terms",
    "Write a Python binary search",
    "Best ML frameworks in 2025?",
    "Prep me for a data science interview",
]

# ──────────────────────────────────────────
# Session State
# ──────────────────────────────────────────
def _init_state():
    for k, v in {
        "sessions":         {"Session 1": {"history": [], "lc_history": [], "count": 0}},
        "active_session":   "Session 1",
        "response_times":   [],
        "last_user_input":  "",
        "current_mode":     None,
        "active_model":     "meta-llama/Llama-3.1-8B-Instruct",
        "ratings":          {},
        "prompt_templates": {
            "Debug my code":    "Please debug the following code and explain each fix:\n\n",
            "Explain concept":  "Explain the concept of [TOPIC] with examples:\n\n",
            "Write unit tests": "Write comprehensive unit tests for:\n\n",
            "Summarize text":   "Summarize the following in 5 bullet points:\n\n",
        },
        "file_context":     "",
        "custom_prompt":    "",
        # ── NEW: RAG state ──────────────────
        "rag_vectorstore":  None,   # FAISS index for the uploaded file
        "rag_file_name":    "",     # name of the indexed file
        "rag_chunk_count":  0,      # how many chunks were indexed
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def sess() -> dict:
    return st.session_state.sessions[st.session_state.active_session]

def chat_history() -> list:
    return sess()["history"]

def lc_history() -> list:
    return sess()["lc_history"]


# ══════════════════════════════════════════════════════════════
# ★  RAG HELPERS  ★
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_embeddings():
    """Load the embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_rag_index(text: str) -> tuple[object, int]:
    """
    Chunk `text` and build a FAISS vector store.
    Returns (vectorstore, chunk_count).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.create_documents([text])
    if not chunks:
        return None, 0

    embeddings = _get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, len(chunks)


def rag_retrieve(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k chunks from the FAISS index most relevant to `query`.
    Returns a formatted context string, or "" if no index exists.
    """
    vs = st.session_state.rag_vectorstore
    if vs is None:
        return ""
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return ""
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    return context


# ══════════════════════════════════════════════════════════════
# ★  WIKIPEDIA RETRIEVER HELPER  ★
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_wiki_retriever(top_k: int = 2, doc_content_chars_max: int = 2000):
    """Return a cached WikipediaRetriever instance."""
    return WikipediaRetriever(
        top_k_results=top_k,
        doc_content_chars_max=doc_content_chars_max,
    )


def wiki_retrieve(query: str, top_k: int = 2) -> str:
    """
    Fetch Wikipedia summaries for `query`.
    Returns a formatted string or "" on failure.
    """
    try:
        retriever = _get_wiki_retriever(top_k=top_k)
        docs = retriever.invoke(query)
        if not docs:
            return ""
        parts = []
        for doc in docs:
            title = doc.metadata.get("title", "Wikipedia")
            parts.append(f"**[Wikipedia: {title}]**\n{doc.page_content[:1500]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"[Wikipedia retrieval error: {e}]"


def build_augmented_prompt_rag(
    query: str,
    rag_ctx: str = "",
    wiki_ctx: str = "",
    web_ctx: str = "",
) -> str:
    """
    Combine all retrieved context into a single augmented prompt.
    Priority order shown to model: RAG file > Wikipedia > Web search.
    """
    sections = []
    if rag_ctx:
        sections.append(
            f"=== RELEVANT EXCERPTS FROM UPLOADED FILE ===\n{rag_ctx}\n=== END FILE EXCERPTS ==="
        )
    if wiki_ctx:
        sections.append(
            f"=== WIKIPEDIA CONTEXT ===\n{wiki_ctx}\n=== END WIKIPEDIA ==="
        )
    if web_ctx:
        sections.append(
            f"=== WEB SEARCH RESULTS ===\n{web_ctx}\n=== END WEB SEARCH ==="
        )

    if not sections:
        return query

    preamble = (
        "You have access to the following retrieved context. "
        "Use it to answer accurately. Do NOT claim you lack information "
        "if it is present below.\n\n"
    )
    return preamble + "\n\n".join(sections) + f"\n\nUser question: {query}\n\nAnswer:"


# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("💬 Chat Sessions")
    session_names = list(st.session_state.sessions.keys())
    active_idx = session_names.index(st.session_state.active_session)
    chosen = st.selectbox("Active Session", session_names, index=active_idx)
    if chosen != st.session_state.active_session:
        st.session_state.active_session = chosen
        st.rerun()

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("➕ New", use_container_width=True):
            n = f"Session {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[n] = {"history": [], "lc_history": [], "count": 0}
            st.session_state.active_session = n
            st.rerun()
    with sc2:
        if st.button("🗑️ Delete", use_container_width=True, disabled=len(session_names) == 1):
            del st.session_state.sessions[st.session_state.active_session]
            st.session_state.active_session = list(st.session_state.sessions.keys())[0]
            st.rerun()

    st.divider()

    st.subheader("🤖 Model")
    category      = st.selectbox("Category", list(MODEL_REGISTRY.keys()))
    models_in_cat = MODEL_REGISTRY[category]
    repo_options  = list(models_in_cat.keys())
    label_options = [models_in_cat[r]["label"] for r in repo_options]
    sel_idx       = st.selectbox("Model", range(len(repo_options)), format_func=lambda i: label_options[i])
    model_name    = repo_options[sel_idx]
    st.session_state.active_model = model_name
    info = ALL_MODELS[model_name]
    st.markdown(
        f"""<div style="background:#1e2130;border-radius:8px;padding:10px;margin:4px 0;">
        <b>{info['label']}</b><br><small>{info['desc']}</small><br><br>
        <small>📏 Context: <b>{info['context']}</b></small><br>
        <small>🏷️ {" · ".join(f"<code>{t}</code>" for t in info['tags'])}</small>
        </div>""", unsafe_allow_html=True)

    st.divider()

    assistant_mode = st.selectbox("Assistant Mode", list(ASSISTANT_MODES))
    if assistant_mode == "Custom":
        custom_input = st.text_area(
            "Write your system prompt", value=st.session_state.custom_prompt,
            height=100, placeholder="You are a helpful assistant that…")
        st.session_state.custom_prompt = custom_input
        ASSISTANT_MODES["Custom"] = custom_input

    response_language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Hinglish", "Spanish", "French", "German", "Japanese"])

    st.divider()
    temperature   = st.slider("Temperature",   0.0, 1.0, 0.5, 0.1)
    max_tokens    = st.slider("Max Tokens",    256, 2048, 1024, 128)
    memory_window = st.slider("Memory Window",   2,   20,   10,   2)

    st.divider()

    # ── Retrieval options ──────────────────
    st.subheader("🔍 Retrieval")

    web_search_enabled  = st.toggle("🌐 Web Search (DuckDuckGo)")
    wiki_search_enabled = st.toggle("📖 Wikipedia Retriever")
    rag_enabled         = st.toggle("📄 RAG (File Search)", value=True,
                                    help="When ON, queries search your uploaded file via vector similarity.")

    if wiki_search_enabled:
        wiki_top_k = st.slider("Wikipedia results", 1, 5, 2)
    else:
        wiki_top_k = 2

    if rag_enabled:
        rag_top_k = st.slider("RAG chunks to retrieve", 2, 8, 4)
    else:
        rag_top_k = 4

    show_retrieval_debug = st.toggle("🔬 Show Retrieval Debug", value=False)

    if web_search_enabled:
        show_search_debug = show_retrieval_debug
    else:
        show_search_debug = False

    st.divider()
    show_timestamps = st.toggle("🕐 Timestamps",      value=True)
    show_stats      = st.toggle("📊 Response Stats",  value=True)
    show_ratings    = st.toggle("⭐ Message Ratings", value=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        sess()["history"]    = []
        sess()["lc_history"] = []
        sess()["count"]      = 0
        st.session_state.last_user_input = ""
        st.session_state.current_mode    = None
        st.session_state.ratings         = {}
        st.rerun()

    if st.session_state.response_times:
        st.divider()
        st.caption("⚡ Session Stats")
        avg_t = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.metric("Avg Response Time", f"{avg_t:.1f}s")
        st.metric("Total Exchanges",   sess()["count"])

    # ── RAG index status ───────────────────
    if st.session_state.rag_vectorstore is not None:
        st.divider()
        st.success(
            f"📚 RAG Index Active\n\n"
            f"**File:** {st.session_state.rag_file_name}\n\n"
            f"**Chunks:** {st.session_state.rag_chunk_count}"
        )
        if st.button("🗑️ Clear RAG Index", use_container_width=True):
            st.session_state.rag_vectorstore = None
            st.session_state.rag_file_name   = ""
            st.session_state.rag_chunk_count  = 0
            st.session_state.file_context     = ""
            st.rerun()

# ──────────────────────────────────────────
# Sync System Prompt
# ──────────────────────────────────────────
system_prompt = ASSISTANT_MODES[assistant_mode]
if response_language != "English":
    system_prompt += f" Always respond in {response_language}."

mode_key = (assistant_mode, response_language, system_prompt)
if st.session_state.current_mode != mode_key:
    st.session_state.current_mode = mode_key
    sys_msg = SystemMessage(content=system_prompt)
    lc = lc_history()
    if lc and isinstance(lc[0], SystemMessage):
        lc[0] = sys_msg
    else:
        lc.insert(0, sys_msg)

# ──────────────────────────────────────────
# Model Loader
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(repo_id: str):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)

# ──────────────────────────────────────────
# Web Search (DuckDuckGo)
# ──────────────────────────────────────────
def search_duckduckgo(query: str, max_results: int = 6) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        if not results:
            return "No results found."
        return "\n\n".join(f"[{r['title']}]\n{r['body']}" for r in results if r.get("body"))
    except Exception as e:
        return f"Search error: {e}"


# ══════════════════════════════════════════════════════════════
# ★★★  RENDERING ENGINE  ★★★
# ══════════════════════════════════════════════════════════════

import html as _html

_BRACKET_DISPLAY_RE = re.compile(r'\\\[([\s\S]*?)\\\]')
_PAREN_INLINE_RE    = re.compile(r'\\\((.*?)\\\)')
_RAW_DISPLAY_RE = re.compile(
    r'(?<!\$)'
    r'(\\begin\{(?:equation|align|gather|multline)\*?'
    r'\}[\s\S]*?\\end\{(?:equation|align|gather|multline)\*?\})',
    re.DOTALL,
)

def _normalise_latex(text: str) -> str:
    text = _BRACKET_DISPLAY_RE.sub(lambda m: f'$$\n{m.group(1).strip()}\n$$', text)
    text = _PAREN_INLINE_RE.sub(lambda m: f'${m.group(1).strip()}$', text)
    text = _RAW_DISPLAY_RE.sub(lambda m: f'$$\n{m.group(1)}\n$$', text)
    return text

_THINK_CLOSED_RE = re.compile(
    r'<(think|thinking|reasoning|scratchpad)>([\s\S]*?)<\/\1>',
    re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    r'<(think|thinking|reasoning|scratchpad)>([\s\S]*)',
    re.IGNORECASE,
)

def _extract_think_blocks(text: str):
    blocks: list[tuple[str, str]] = []
    def _replace_closed(m):
        blocks.append((m.group(1).capitalize(), m.group(2).strip()))
        return ""
    text = _THINK_CLOSED_RE.sub(_replace_closed, text).strip()
    m = _THINK_OPEN_RE.search(text)
    if m:
        blocks.append((m.group(1).capitalize(), m.group(2).strip()))
        text = text[: m.start()].strip()
    return text, blocks

CALLOUT_MAP = {
    "warning":("warning","⚠️"), "tip":("tip","💡"),
    "info":("info","ℹ️"),       "note":("info","📝"),
    "success":("success","✅"), "error":("error","❌"),
    "danger":("error","🚨"),
}
_EMOJI_TO_KEY = {"⚠️":"warning","💡":"tip","ℹ️":"info","✅":"success","❌":"error","🚨":"error"}

_CALLOUT_LINE_RE = re.compile(
    r'^>\s*(?:(⚠️|💡|ℹ️|✅|❌|🚨)\s*)?'
    r'(?:\*\*)?(?:(warning|tip|info|note|success|error|danger))(?:\*\*)?\s*[:\-]?\s*(.*)',
    re.IGNORECASE)

def _try_callout(line: str):
    m = _CALLOUT_LINE_RE.match(line)
    if not m:
        return None
    emoji_raw = (m.group(1) or "").strip()
    label_raw = (m.group(2) or "").strip().lower()
    body      = (m.group(3) or "").strip()
    key = label_raw or _EMOJI_TO_KEY.get(emoji_raw, "")
    cls, icon = CALLOUT_MAP.get(key, ("info", "ℹ️"))
    return (f'<div class="callout {cls}">'
            f'<span class="callout-icon">{icon}</span>'
            f'<span>{_html.escape(body)}</span></div>')

_TABLE_ROW_RE  = re.compile(r'^\s*\|.*\|\s*$')
_TABLE_SEP_RE  = re.compile(r'^\s*\|[\s\|\-:]+\|\s*$')

def _wrap_bare_latex_cell(cell: str) -> str:
    cell = cell.strip()
    if '$' in cell:
        return cell
    if re.search(r'\\[a-zA-Z]', cell):
        return f'${cell}$'
    return cell

def _sanitise_table_block(table_lines: list[str]) -> str:
    out = []
    for line in table_lines:
        line = re.sub(r'<br\s*/?>', ' ', line, flags=re.IGNORECASE)
        if line.strip().startswith('$$') and line.strip().endswith('$$'):
            out.append(line)
            continue
        if _TABLE_SEP_RE.match(line):
            out.append(line)
            continue
        if _TABLE_ROW_RE.match(line):
            parts = line.split('|')
            cells = [_wrap_bare_latex_cell(p) for p in parts]
            out.append('|'.join(cells))
        else:
            out.append(line)
    return '\n'.join(out)

def _preprocess_tables(text: str) -> str:
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_ROW_RE.match(line):
            table_block = []
            while i < len(lines) and (
                _TABLE_ROW_RE.match(lines[i]) or
                _TABLE_SEP_RE.match(lines[i])
            ):
                table_block.append(lines[i])
                i += 1
            result.append(_sanitise_table_block(table_block))
        else:
            result.append(line)
            i += 1
    return '\n'.join(result)

def _try_render_data(code: str, lang: str) -> bool:
    import pandas as pd
    if lang.lower() == "json":
        try:
            data = json.loads(code)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                st.dataframe(pd.DataFrame(data), use_container_width=True)
                return True
        except Exception:
            pass
    if lang.lower() == "csv":
        try:
            df = pd.read_csv(io.StringIO(code))
            st.dataframe(df, use_container_width=True)
            return True
        except Exception:
            pass
    return False

_SPLIT_RE = re.compile(
    r'(```[\w]*\n?[\s\S]*?```'
    r'|\$\$[\s\S]*?\$\$'
    r')',
    re.DOTALL,
)

_LANG_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
}

def _canon_lang(lang: str) -> str:
    return _LANG_ALIASES.get(lang.lower(), lang.lower())

def render_response(text: str):
    text = _normalise_latex(text)
    text = _preprocess_tables(text)
    text, think_blocks = _extract_think_blocks(text)
    for label, content in think_blocks:
        st.markdown(
            f'<details class="think-block">'
            f'<summary>Chain of Thought ({label}) — click to expand</summary>'
            f'<div class="think-body">{_html.escape(content)}</div>'
            f'</details>',
            unsafe_allow_html=True)
    if not text.strip():
        return
    parts = _SPLIT_RE.split(text)
    for part in parts:
        if not part.strip():
            continue
        if part.startswith("```"):
            m = re.match(r'```(\w*)\n?([\s\S]*?)```', part, re.DOTALL)
            if not m:
                st.code(part)
                continue
            lang = _canon_lang(m.group(1).strip())
            code = m.group(2)
            if lang in ("json", "csv") and _try_render_data(code, lang):
                continue
            st.code(code, language=lang if lang else None)
        elif part.startswith("$$") and part.endswith("$$"):
            formula = part[2:-2].strip()
            if formula:
                try:
                    st.latex(formula)
                except Exception:
                    st.markdown(part)
        else:
            lines, plain_batch = part.split("\n"), []
            def _flush():
                nonlocal plain_batch
                if plain_batch:
                    st.markdown("\n".join(plain_batch))
                    plain_batch = []
            for line in lines:
                callout = _try_callout(line)
                if callout:
                    _flush()
                    st.markdown(callout, unsafe_allow_html=True)
                else:
                    plain_batch.append(line)
            _flush()

def render_streaming_chunk(text: str, placeholder):
    normalised = _normalise_latex(text)
    normalised = re.sub(r'<br\s*/?>', ' ', normalised, flags=re.IGNORECASE)
    normalised = re.sub(
        r'<(think|thinking|reasoning|scratchpad)>[\s\S]*$',
        '', normalised, flags=re.IGNORECASE
    ).strip()
    normalised = re.sub(r'```[\w]*\n?(?![\s\S]*```)', '', normalised).strip()
    placeholder.markdown(normalised + " ▌")


# ──────────────────────────────────────────
# File Reader  (now also builds RAG index)
# ──────────────────────────────────────────
def extract_file_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".txt", ".md", ".py", ".csv")):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                return "[Install PyPDF2: pip install PyPDF2]"
        elif name.endswith(".json"):
            data = json.load(uploaded_file)
            return json.dumps(data, indent=2)
        else:
            return f"[Unsupported file type: {name}]"
    except Exception as e:
        return f"[File read error: {e}]"


# ──────────────────────────────────────────
# History Helpers
# ──────────────────────────────────────────
def add_message(role: str, content: str):
    ts = datetime.now().strftime("%H:%M:%S")
    sess()["history"].append((role, content, ts))
    cls = HumanMessage if role == "user" else AIMessage
    lc_history().append(cls(content=content))

def get_context() -> list:
    lc       = lc_history()
    sys_msgs = [m for m in lc if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in lc if not isinstance(m, SystemMessage)]
    return sys_msgs + conv_msgs[-memory_window:]


# ──────────────────────────────────────────
# Header
# ──────────────────────────────────────────
cur_info = ALL_MODELS[model_name]
st.title("🤖 AI Chatbot")
st.caption(
    f"**{cur_info['label']}** · Mode: {assistant_mode} · "
    f"Lang: {response_language} · Session: **{st.session_state.active_session}**")

# ──────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────
tab_chat, tab_file, tab_templates = st.tabs(["💬 Chat", "📁 File Upload & RAG", "📋 Prompt Templates"])

# ══════════════════════════════════════════
# TAB: File Upload  →  now builds RAG index
# ══════════════════════════════════════════
with tab_file:
    st.subheader("📁 Upload a File — RAG Indexing Enabled")
    st.caption("Supported: .txt, .md, .py, .csv, .json, .pdf")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.info("**📄 RAG Mode**\nFile is chunked & embedded into a FAISS vector store. Only the most relevant chunks are sent to the model per query.")
    col_info2.info("**📖 Wikipedia Mode**\nEnable in sidebar → the model also gets live Wikipedia summaries for your query.")
    col_info3.info("**🌐 Web Mode**\nEnable DuckDuckGo in sidebar → web results are also injected as context.")

    uploaded = st.file_uploader(
        "Upload file", type=["txt","md","py","csv","json","pdf"],
        label_visibility="collapsed")

    if uploaded:
        extracted = extract_file_text(uploaded)

        # ── Build RAG index ────────────────
        with st.spinner("🔨 Building RAG index (chunking + embedding)…"):
            vs, n_chunks = build_rag_index(extracted)

        if vs is not None:
            st.session_state.rag_vectorstore = vs
            st.session_state.rag_file_name   = uploaded.name
            st.session_state.rag_chunk_count  = n_chunks
            st.session_state.file_context     = extracted  # kept for fallback / export
            st.success(
                f"✅ RAG index built for **{uploaded.name}** — "
                f"{n_chunks} chunks · {len(extracted):,} characters"
            )
        else:
            st.warning("⚠️ Could not build RAG index (empty file?). Falling back to full-text context.")
            st.session_state.file_context = extracted

        with st.expander("Preview file content"):
            st.code(extracted[:2000], language="text")

    # ── RAG test panel ─────────────────────
    if st.session_state.rag_vectorstore is not None:
        st.divider()
        st.subheader("🔍 Test RAG Retrieval")
        test_q = st.text_input("Enter a test query to see which chunks get retrieved:")
        if test_q:
            with st.spinner("Retrieving…"):
                result = rag_retrieve(test_q, k=rag_top_k if rag_enabled else 4)
            if result:
                st.markdown("**Retrieved chunks:**")
                st.markdown(result)
            else:
                st.info("No chunks retrieved.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(
                f"📎 **{st.session_state.rag_file_name}** active\n\n"
                f"{st.session_state.rag_chunk_count} chunks indexed"
            )
        with col_b:
            if st.button("❌ Remove File & Index"):
                st.session_state.rag_vectorstore = None
                st.session_state.rag_file_name   = ""
                st.session_state.rag_chunk_count  = 0
                st.session_state.file_context     = ""
                st.rerun()

# ══════════════════════════════════════════
# TAB: Prompt Templates
# ══════════════════════════════════════════
with tab_templates:
    st.subheader("📋 Prompt Templates")
    st.caption("Save and reuse your favourite prompts.")
    with st.expander("➕ Save a new template"):
        t_name = st.text_input("Template name")
        t_body = st.text_area("Template text", height=100)
        if st.button("Save Template") and t_name and t_body:
            st.session_state.prompt_templates[t_name] = t_body
            st.success(f"Saved: {t_name}")
    st.divider()
    for tname, tbody in list(st.session_state.prompt_templates.items()):
        col1, col2, col3 = st.columns([4, 1, 1])
        col1.markdown(f"**{tname}**")
        col1.caption(tbody[:80] + "…" if len(tbody) > 80 else tbody)
        if col2.button("Use", key=f"use_{tname}"):
            st.session_state["pending_template"] = tbody
            st.rerun()
        if col3.button("🗑️", key=f"del_{tname}"):
            del st.session_state.prompt_templates[tname]
            st.rerun()

# ══════════════════════════════════════════
# TAB: Chat
# ══════════════════════════════════════════
with tab_chat:

    cols = st.columns(len(SUGGESTIONS))
    selected_prompt = None
    for col, s in zip(cols, SUGGESTIONS):
        if col.button(s, use_container_width=True):
            selected_prompt = s

    st.divider()

    # ── Active retrieval indicators ────────
    indicators = []
    if rag_enabled and st.session_state.rag_vectorstore is not None:
        indicators.append(f'<span class="rag-badge">📄 RAG: {st.session_state.rag_file_name}</span>')
    if wiki_search_enabled:
        indicators.append(f'<span class="wiki-badge">📖 Wikipedia ON</span>')
    if web_search_enabled:
        indicators.append(f'<span class="stat-badge">🌐 Web Search ON</span>')
    if indicators:
        st.markdown(" ".join(indicators), unsafe_allow_html=True)
        st.markdown("")

    # ── Chat History ───────────────────────
    for i, (role, content, ts) in enumerate(chat_history()):
        with st.chat_message(role):
            render_response(content)
            row = st.columns([6, 1, 1, 1])
            if show_timestamps:
                row[0].caption(ts)
            if role == "assistant":
                with st.expander("📋 Copy raw", expanded=False):
                    st.code(content, language="markdown")
            if show_ratings and role == "assistant":
                rating_key = f"rating_{st.session_state.active_session}_{i}"
                current    = st.session_state.ratings.get(rating_key, None)
                with row[2]:
                    if st.button("👍", key=f"up_{i}", help="Good response",
                                 type="primary" if current == "👍" else "secondary"):
                        st.session_state.ratings[rating_key] = "👍"
                        st.rerun()
                with row[3]:
                    if st.button("👎", key=f"dn_{i}", help="Bad response",
                                 type="primary" if current == "👎" else "secondary"):
                        st.session_state.ratings[rating_key] = "👎"
                        st.rerun()

    # ── Input ──────────────────────────────
    pending_template = st.session_state.pop("pending_template", None)
    user_input       = st.chat_input("Type your message…")
    if selected_prompt:  user_input = selected_prompt
    if pending_template: user_input = pending_template

    col_regen, col_summary, col_auto_title = st.columns(3)
    with col_regen:
        if st.button("🔄 Regenerate", disabled=not st.session_state.last_user_input):
            h  = chat_history()
            lc = lc_history()
            if h  and h[-1][0]           == "assistant": h.pop()
            if lc and isinstance(lc[-1], AIMessage):     lc.pop()
            user_input = st.session_state.last_user_input
    with col_summary:
        summarize  = st.button("📝 Summarize Chat",    disabled=len(chat_history()) < 4)
    with col_auto_title:
        auto_title = st.button("🏷️ Auto-Title Session", disabled=len(chat_history()) < 2)

    if auto_title:
        full_text = " ".join(c for _, c, _ in chat_history()[:4])
        model_obj = load_model(model_name)
        with st.spinner("Generating title…"):
            resp = model_obj.invoke([HumanMessage(
                content=f"Generate a 4-5 word title for this conversation. "
                        f"Only output the title, nothing else:\n\n{full_text[:500]}")])
            new_title = resp.content.strip().strip('"').strip("'")[:40]
            old = st.session_state.active_session
            st.session_state.sessions[new_title] = st.session_state.sessions.pop(old)
            st.session_state.active_session = new_title
            st.rerun()

    if summarize:
        full_text = "\n".join(f"{r.upper()}: {c}" for r, c, _ in chat_history())
        model_obj = load_model(model_name)
        with st.expander("📝 Conversation Summary", expanded=True):
            with st.spinner("Summarizing…"):
                resp = model_obj.invoke([HumanMessage(
                    content=f"Summarize in 5 bullets:\n\n{full_text}")])
                render_response(resp.content)

    # ── Main Chat Logic ────────────────────────────────────────
    if user_input:
        raw = user_input.strip()
        st.session_state.last_user_input = raw

        # ── Step 1: RAG retrieval ──────────
        rag_ctx = ""
        if rag_enabled and st.session_state.rag_vectorstore is not None:
            with st.spinner("📄 Searching uploaded file (RAG)…"):
                rag_ctx = rag_retrieve(raw, k=rag_top_k)

        # ── Step 2: Wikipedia retrieval ────
        wiki_ctx = ""
        if wiki_search_enabled:
            with st.spinner("📖 Fetching Wikipedia context…"):
                wiki_ctx = wiki_retrieve(raw, top_k=wiki_top_k)

        # ── Step 3: Web search ─────────────
        web_ctx = ""
        if web_search_enabled:
            with st.spinner("🌐 Searching the web…"):
                web_ctx = search_duckduckgo(raw)

        # ── Step 4: Build final prompt ─────
        if rag_ctx or wiki_ctx or web_ctx:
            lc_input = build_augmented_prompt_rag(raw, rag_ctx, wiki_ctx, web_ctx)
        else:
            lc_input = raw

        add_message("user", raw)
        # Replace last HumanMessage with the augmented version if needed
        if lc_input != raw:
            lc_history()[-1] = HumanMessage(content=lc_input)

        with st.chat_message("user"):
            st.markdown(raw)

        # ── Debug panels ───────────────────
        if show_retrieval_debug:
            if rag_ctx:
                with st.expander("📄 RAG Chunks Retrieved", expanded=False):
                    st.markdown(rag_ctx)
            if wiki_ctx:
                with st.expander("📖 Wikipedia Context Retrieved", expanded=False):
                    st.markdown(wiki_ctx)
            if web_ctx:
                with st.expander("🌐 Web Search Results", expanded=False):
                    st.text(web_ctx)

        model_obj = load_model(model_name)
        ctx       = get_context()

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            t0 = time.time()

            try:
                for chunk in model_obj.stream(ctx):
                    if chunk.content:
                        full_response += chunk.content
                        render_streaming_chunk(full_response, placeholder)

                placeholder.empty()
                render_response(full_response)

            except Exception as e:
                full_response = f"⚠️ Error: {e}"
                placeholder.error(full_response)

            elapsed = round(time.time() - t0, 2)
            st.session_state.response_times.append(elapsed)

            # ── Source badges ───────────────
            source_badges = []
            if rag_ctx:
                source_badges.append(f'<span class="rag-badge">📄 RAG</span>')
            if wiki_ctx:
                source_badges.append(f'<span class="wiki-badge">📖 Wikipedia</span>')
            if web_ctx:
                source_badges.append(f'<span class="stat-badge">🌐 Web</span>')

            if show_stats:
                words      = len(full_response.split())
                est_tokens = len(full_response) // 4
                badges = (
                    f'<span class="stat-badge">⏱ {elapsed}s</span>'
                    f'<span class="stat-badge">📝 {words} words</span>'
                    f'<span class="stat-badge">🔢 ~{est_tokens} tokens</span>'
                    f'<span class="stat-badge">🤖 {cur_info["label"]}</span>'
                    + ("  " + " ".join(source_badges) if source_badges else "")
                )
                st.markdown(badges, unsafe_allow_html=True)

            with st.expander("📋 Copy raw response", expanded=False):
                st.code(full_response, language="markdown")

        add_message("assistant", full_response)
        sess()["count"] += 1

    # ── Export ─────────────────────────────
    st.divider()
    with st.expander("📥 Export Chat"):
        if not chat_history():
            st.info("No messages yet.")
        else:
            ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
            lines  = [f"# {st.session_state.active_session} — {ts_now}\nModel: {cur_info['label']}\n"]
            for role, content, ts in chat_history():
                label = "**User**" if role == "user" else "**Assistant**"
                lines.append(f"{label} _{ts}_\n\n{content}\n")
            md_export   = "\n---\n".join(lines)
            txt_export  = "\n".join(f"[{ts}] {r.upper()}: {c}" for r, c, ts in chat_history())
            json_export = json.dumps([
                {"role": r, "content": c, "timestamp": ts,
                 "rating": st.session_state.ratings.get(
                     f"rating_{st.session_state.active_session}_{i}", None)}
                for i, (r, c, ts) in enumerate(chat_history())
            ], indent=2)

            c1, c2, c3 = st.columns(3)
            c1.download_button("📄 Markdown", md_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown", use_container_width=True)
            c2.download_button("📃 Plain Text", txt_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True)
            c3.download_button("🗂️ JSON", json_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True)

            all_ratings = list(st.session_state.ratings.values())
            if all_ratings:
                st.divider()
                st.caption(f"⭐ Ratings — 👍 {all_ratings.count('👍')}  ·  👎 {all_ratings.count('👎')}")

st.caption("Built with Streamlit · LangChain · Hugging Face · FAISS · Wikipedia")