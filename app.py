import time
import json
import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ── KaTeX for LaTeX + minimal table/think styles ──────────────
st.markdown(r"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="
    renderMathInElement(document.body,{
      delimiters:[
        {left:'$$',right:'$$',display:true},
        {left:'$',right:'$',display:false},
        {left:'\\[',right:'\\]',display:true},
        {left:'\\(',right:'\\)',display:false}
      ],throwOnError:false});
    new MutationObserver(function(){
      renderMathInElement(document.body,{
        delimiters:[
          {left:'$$',right:'$$',display:true},
          {left:'$',right:'$',display:false},
          {left:'\\[',right:'\\]',display:true},
          {left:'\\(',right:'\\)',display:false}
        ],throwOnError:false});
    }).observe(document.body,{childList:true,subtree:true});
  "></script>
<style>
.katex { font-size: 1.1rem !important }
.katex-display { overflow-x: auto; padding: .4rem 0 }
table { width:100%!important; border-collapse:collapse!important; margin:.8rem 0!important }
thead tr { background:#1a3a5c!important; color:#e6edf3!important; font-weight:700!important }
th, td { padding:8px 12px!important; text-align:left!important;
         border-bottom:1px solid #21262d!important; border-right:1px solid #21262d!important }
th:last-child, td:last-child { border-right:none!important }
tbody tr:nth-child(even) { background:#161b22!important }
details.think-block { background:#0d1117; border:1px solid #30363d;
                       border-radius:8px; margin:.6rem 0 }
details.think-block summary { padding:.5rem .8rem; cursor:pointer; color:#8b949e;
                               font-size:.82rem; font-style:italic; list-style:none }
details.think-block summary::before { content:"🧠 " }
details.think-block .think-body { padding:.6rem 1rem .8rem; color:#8b949e;
  font-size:.88rem; border-top:1px solid #21262d; font-style:italic;
  line-height:1.6; white-space:pre-wrap }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Model Registry  (Llama + Qwen only)
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "🦙 Llama": {
        "meta-llama/Llama-3.1-8B-Instruct": {
            "label": "Llama 3.1 8B", "context": "128k",
            "desc": "Meta's fast open chat model."},
    },
    "🧠 Qwen": {
        "Qwen/Qwen2.5-7B-Instruct": {
            "label": "Qwen 2.5 7B", "context": "128k",
            "desc": "Fast, multilingual, great at reasoning."},
        "Qwen/Qwen2.5-72B-Instruct": {
            "label": "Qwen 2.5 72B ⭐", "context": "128k",
            "desc": "Near GPT-4 quality — best free model."},
        "Qwen/Qwen2.5-Coder-32B-Instruct": {
            "label": "Qwen Coder 32B ⭐", "context": "128k",
            "desc": "Best free coding model on HF."},
    },
}

ALL_MODELS: dict[str, dict] = {}
for _cat, _models in MODEL_REGISTRY.items():
    for _repo, _info in _models.items():
        ALL_MODELS[_repo] = {**_info, "category": _cat}

# ──────────────────────────────────────────
# Assistant Modes
# ──────────────────────────────────────────
ASSISTANT_MODES = {
    "General": (
        "You are a helpful, concise AI assistant. "
        "Format math with LaTeX: inline $expr$, block $$expr$$. "
        "Format code in fenced blocks with language tag. "
        "Use markdown tables for comparisons."
    ),
    "Python Expert": (
        "You are a senior Python developer. "
        "Write clean, production-quality, well-commented code. "
        "Always use fenced code blocks with ```python. "
        "Format equations with LaTeX."
    ),
    "Data Scientist": (
        "You are an expert in ML, Data Science, AI and GenAI. "
        "ALWAYS render math using LaTeX: $$...$$ for display, $...$ for inline. "
        "Format code with proper language tags. Use markdown tables."
    ),
    "Socratic Tutor": (
        "Guide users through questions so they discover answers themselves. "
        "Use LaTeX for equations and numbered steps for reasoning."
    ),
    "Custom": "",
}

SUGGESTIONS = [
    "Explain transformers in simple terms",
    "Write a Python binary search",
    "Best ML frameworks in 2025?",
    "Solve: integral of x^2 dx",
]

# ──────────────────────────────────────────
# Session State
# ──────────────────────────────────────────
def _init_state():
    defaults = {
        "sessions":        {"Session 1": {"history": [], "lc_history": [], "count": 0}},
        "active_session":  "Session 1",
        "response_times":  [],
        "last_user_input": "",
        "current_mode":    None,
        "active_model":    "meta-llama/Llama-3.1-8B-Instruct",
        "custom_prompt":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def sess():
    return st.session_state.sessions[st.session_state.active_session]

def chat_history():
    return sess()["history"]

def lc_history():
    return sess()["lc_history"]

# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Sessions
    st.subheader("💬 Sessions")
    session_names = list(st.session_state.sessions.keys())
    chosen = st.selectbox("Active Session",
                          session_names,
                          index=session_names.index(st.session_state.active_session))
    if chosen != st.session_state.active_session:
        st.session_state.active_session = chosen
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New", use_container_width=True):
            n = f"Session {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[n] = {"history": [], "lc_history": [], "count": 0}
            st.session_state.active_session = n
            st.rerun()
    with c2:
        if st.button("🗑️ Delete", use_container_width=True, disabled=len(session_names) == 1):
            del st.session_state.sessions[st.session_state.active_session]
            st.session_state.active_session = list(st.session_state.sessions.keys())[0]
            st.rerun()

    st.divider()

    # Model
    st.subheader("🤖 Model")
    category      = st.selectbox("Category", list(MODEL_REGISTRY.keys()))
    models_in_cat = MODEL_REGISTRY[category]
    repo_options  = list(models_in_cat.keys())
    label_options = [models_in_cat[r]["label"] for r in repo_options]
    sel_idx    = st.selectbox("Model", range(len(repo_options)),
                              format_func=lambda i: label_options[i])
    model_name = repo_options[sel_idx]
    st.session_state.active_model = model_name
    info = ALL_MODELS[model_name]
    st.caption(f"**{info['label']}** · {info['desc']} · Context: {info['context']}")

    st.divider()

    # Mode & Language
    assistant_mode = st.selectbox("Assistant Mode", list(ASSISTANT_MODES))
    if assistant_mode == "Custom":
        custom_input = st.text_area("System prompt", value=st.session_state.custom_prompt,
                                    height=100, placeholder="You are a helpful assistant…")
        st.session_state.custom_prompt = custom_input
        ASSISTANT_MODES["Custom"] = custom_input

    response_language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Hinglish", "Spanish", "French", "German", "Japanese"])

    st.divider()

    # Generation params
    temperature   = st.slider("Temperature",   0.0, 1.0, 0.5, 0.1)
    max_tokens    = st.slider("Max Tokens",    256, 2048, 1024, 128)
    memory_window = st.slider("Memory Window",   2,   20,   10,   2)

    st.divider()

    # Web search
    web_search_enabled = st.toggle("🌐 Web Search")
    show_search_debug  = False
    if web_search_enabled:
        search_provider   = st.selectbox("Provider", ["DuckDuckGo", "Wikipedia", "Both"])
        show_search_debug = st.toggle("Show Search Results")
    else:
        search_provider = "DuckDuckGo"

    st.divider()

    show_timestamps = st.toggle("🕐 Timestamps", value=True)
    show_stats      = st.toggle("📊 Response Stats", value=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        sess()["history"]    = []
        sess()["lc_history"] = []
        sess()["count"]      = 0
        st.session_state.last_user_input = ""
        st.session_state.current_mode    = None
        st.rerun()

    if st.session_state.response_times:
        st.divider()
        avg_t = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.metric("Avg Response", f"{avg_t:.1f}s")
        st.metric("Exchanges",    sess()["count"])

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
def load_model(repo_id: str, temp: float, max_new_tokens: int):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, task="text-generation",
        max_new_tokens=max_new_tokens, temperature=temp)
    return ChatHuggingFace(llm=llm)

# ──────────────────────────────────────────
# Web Search
# ──────────────────────────────────────────
def _search_ddg(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=6)
        return "\n\n".join(
            f"[{r['title']}]\n{r['body']}" for r in results if r.get("body")
        ) or "No results found."
    except Exception as e:
        return f"DuckDuckGo error: {e}"

def _search_wiki(query: str) -> str:
    try:
        import wikipedia
        pages = wikipedia.search(query)
        if not pages:
            return "No Wikipedia results."
        return f"[Wikipedia: {pages[0]}]\n{wikipedia.summary(pages[0], sentences=5)}"
    except Exception as e:
        return f"Wikipedia error: {e}"

def do_web_search(query: str) -> str:
    if search_provider == "Wikipedia": return _search_wiki(query)
    if search_provider == "Both":      return _search_wiki(query) + "\n\n" + _search_ddg(query)
    return _search_ddg(query)

def augment_prompt(query: str, results: str) -> str:
    return (
        "You have access to LIVE web search results. "
        "Use them to answer. Do NOT claim you lack current info.\n\n"
        f"=== WEB SEARCH RESULTS ===\n{results}\n=== END ===\n\n"
        f"User question: {query}\n\nAnswer using the search results:"
    )

# ──────────────────────────────────────────
# LaTeX Normaliser  (handles Qwen bare LaTeX)
# ──────────────────────────────────────────
_RE_BRACKET  = re.compile(r'\\\[([\s\S]*?)\\\]')
_RE_PAREN    = re.compile(r'\\\((.*?)\\\)')
_RE_ENV      = re.compile(
    r'(?<!\$)(\\begin\{(?:equation|align|gather|multline)\*?'
    r'\}[\s\S]*?\\end\{(?:equation|align|gather|multline)\*?\})',
    re.DOTALL)
_RE_BOXED    = re.compile(r'(?<!\$)(\\boxed\{[^}]*\})(?!\$)')

def _normalise_latex(text: str) -> str:
    text = _RE_BRACKET.sub(lambda m: f'$$\n{m.group(1).strip()}\n$$', text)
    text = _RE_PAREN.sub(  lambda m: f'${m.group(1).strip()}$',        text)
    text = _RE_ENV.sub(    lambda m: f'$$\n{m.group(1)}\n$$',          text)
    text = _RE_BOXED.sub(  lambda m: f'$${m.group(1)}$$',              text)
    return text

# ──────────────────────────────────────────
# Think-block extractor
# ──────────────────────────────────────────
_RE_THINK_CLOSED = re.compile(
    r'<(think|thinking|reasoning|scratchpad)>([\s\S]*?)<\/\1>', re.IGNORECASE)
_RE_THINK_OPEN   = re.compile(
    r'<(think|thinking|reasoning|scratchpad)>([\s\S]*)',        re.IGNORECASE)

def _extract_think(text: str):
    blocks = []
    def _sub(m):
        blocks.append((m.group(1).capitalize(), m.group(2).strip()))
        return ""
    text = _RE_THINK_CLOSED.sub(_sub, text).strip()
    m = _RE_THINK_OPEN.search(text)
    if m:
        blocks.append((m.group(1).capitalize(), m.group(2).strip()))
        text = text[:m.start()].strip()
    return text, blocks

# ──────────────────────────────────────────
# Markdown table → rendered via components.html
# (ensures KaTeX renders $…$ inside cells)
# ──────────────────────────────────────────
import html as _html

def _render_table(tbl_lines: list[str]):
    """Render a markdown table with full KaTeX support inside cells."""
    headers = [c.strip() for c in tbl_lines[0].split("|") if c.strip()]
    rows = []
    for row_line in tbl_lines[2:]:
        cells = [c.strip() for c in row_line.split("|") if c.strip()]
        if cells:
            rows.append(cells)

    th_html = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    tbody_html = ""
    for cells in rows:
        # Do NOT escape cell content — KaTeX needs raw $...$ to render math
        tbody_html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    full_html = f"""
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{
    delimiters:[
      {{left:'$$',right:'$$',display:true}},
      {{left:'$',right:'$',display:false}}
    ],throwOnError:false}});"></script>
<style>
  body {{ margin:0; padding:0; background:transparent; font-family: sans-serif; font-size:14px; color:#e6edf3 }}
  table {{ width:100%; border-collapse:collapse; }}
  thead tr {{ background:#1a3a5c; color:#e6edf3; font-weight:700 }}
  th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #21262d; border-right:1px solid #21262d }}
  th:last-child, td:last-child {{ border-right:none }}
  tbody tr:nth-child(even) {{ background:#161b22 }}
  tbody tr:hover {{ background:#1c2736 }}
  .katex {{ font-size:1rem !important }}
</style>
</head>
<body>
<table>
  <thead><tr>{th_html}</tr></thead>
  <tbody>{tbody_html}</tbody>
</table>
</body>
</html>"""

    row_count = len(rows) + 1  # +1 for header
    height = row_count * 42 + 20
    components.html(full_html, height=height, scrolling=False)


def _has_table(text: str) -> bool:
    return any("|" in l and l.strip().startswith("|") for l in text.split("\n"))


def _split_tables(text: str):
    """Yield (is_table, lines_or_text) chunks."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if "|" in lines[i] and lines[i].strip().startswith("|"):
            tbl = []
            while i < len(lines) and "|" in lines[i]:
                tbl.append(lines[i]); i += 1
            yield (True, tbl)
        else:
            prose = []
            while i < len(lines) and not ("|" in lines[i] and lines[i].strip().startswith("|")):
                prose.append(lines[i]); i += 1
            yield (False, "\n".join(prose))

# ──────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────
_SPLIT_RE = re.compile(r'(```[\w]*\n?[\s\S]*?```|\$\$[\s\S]*?\$\$)', re.DOTALL)

def render_response(text: str):
    text = _normalise_latex(text)
    text, think_blocks = _extract_think(text)

    for label, content in think_blocks:
        st.markdown(
            f'<details class="think-block">'
            f'<summary>Chain of Thought ({label}) — click to expand</summary>'
            f'<div class="think-body">{_html.escape(content)}</div>'
            f'</details>',
            unsafe_allow_html=True)

    if not text.strip():
        return

    for part in _SPLIT_RE.split(text):
        if not part.strip():
            continue

        # Fenced code block — use native st.code() (no HTML injection)
        if part.startswith("```"):
            m = re.match(r'```(\w*)\n?([\s\S]*?)```', part, re.DOTALL)
            if m:
                st.code(m.group(2), language=m.group(1).strip() or None)
            else:
                st.code(part)

        # Display math $$…$$
        elif part.startswith("$$") and part.endswith("$$"):
            formula = part[2:-2].strip()
            if formula:
                try:
                    st.latex(formula)
                except Exception:
                    st.markdown(part, unsafe_allow_html=True)

        # Prose — split out tables and render each piece correctly
        else:
            for is_table, chunk in _split_tables(part):
                if is_table and len(chunk) >= 2:
                    _render_table(chunk)
                elif chunk.strip():
                    st.markdown(
                        f'<div style="line-height:1.75">{chunk}</div>',
                        unsafe_allow_html=True)


def render_streaming(text: str, placeholder):
    """Lightweight streaming preview."""
    normalised = _normalise_latex(text)
    normalised = re.sub(
        r'<(think|thinking|reasoning|scratchpad)>[\s\S]*$',
        '', normalised, flags=re.IGNORECASE).strip()
    placeholder.markdown(normalised + " ▌", unsafe_allow_html=True)

# ──────────────────────────────────────────
# History helpers
# ──────────────────────────────────────────
def add_message(role: str, content: str):
    ts = datetime.now().strftime("%H:%M:%S")
    sess()["history"].append((role, content, ts))
    lc_history().append((HumanMessage if role == "user" else AIMessage)(content=content))

def get_context():
    lc  = lc_history()
    sys = [m for m in lc if isinstance(m, SystemMessage)]
    conv = [m for m in lc if not isinstance(m, SystemMessage)]
    return sys + conv[-memory_window:]

# ──────────────────────────────────────────
# Header
# ──────────────────────────────────────────
cur_info = ALL_MODELS[model_name]
st.title("🤖 AI Chatbot")
st.caption(
    f"**{cur_info['label']}** · Mode: {assistant_mode} · "
    f"Lang: {response_language} · Session: **{st.session_state.active_session}**")

# ── Suggestion buttons ─────────────────────
cols = st.columns(len(SUGGESTIONS))
selected_prompt = None
for col, s in zip(cols, SUGGESTIONS):
    if col.button(s, use_container_width=True):
        selected_prompt = s

st.divider()

# ──────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────
for role, content, ts in chat_history():
    with st.chat_message(role):
        render_response(content)
        if show_timestamps:
            st.caption(ts)

# ──────────────────────────────────────────
# Input
# ──────────────────────────────────────────
user_input = st.chat_input("Type your message…")
if selected_prompt:
    user_input = selected_prompt

if st.button("🔄 Regenerate", disabled=not st.session_state.last_user_input):
    h, lc = chat_history(), lc_history()
    if h  and h[-1][0]           == "assistant": h.pop()
    if lc and isinstance(lc[-1], AIMessage):     lc.pop()
    user_input = st.session_state.last_user_input

# ──────────────────────────────────────────
# Chat logic
# ──────────────────────────────────────────
if user_input:
    raw = user_input.strip()
    st.session_state.last_user_input = raw

    search_results = None
    if web_search_enabled:
        with st.spinner(f"🌐 Searching {search_provider}…"):
            search_results = do_web_search(raw)
        lc_input = augment_prompt(raw, search_results)
    else:
        lc_input = raw

    add_message("user", raw)
    if web_search_enabled:
        lc_history()[-1] = HumanMessage(content=lc_input)

    with st.chat_message("user"):
        st.markdown(raw)

    if web_search_enabled and show_search_debug and search_results:
        with st.expander("🔍 Search Results"):
            st.text(search_results)

    model_obj = load_model(model_name, temperature, max_tokens)
    ctx = get_context()

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        t0 = time.time()

        try:
            for chunk in model_obj.stream(ctx):
                if chunk.content:
                    full_response += chunk.content
                    render_streaming(full_response, placeholder)
            placeholder.empty()
            render_response(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.error(full_response)

        elapsed = round(time.time() - t0, 2)
        st.session_state.response_times.append(elapsed)

        if show_stats:
            st.caption(
                f"⏱ {elapsed}s · 📝 {len(full_response.split())} words · 🤖 {cur_info['label']}")

    add_message("assistant", full_response)
    sess()["count"] += 1

# ──────────────────────────────────────────
# Export
# ──────────────────────────────────────────
st.divider()
with st.expander("📥 Export Chat"):
    if not chat_history():
        st.info("No messages yet.")
    else:
        ts_now   = datetime.now().strftime("%Y-%m-%d %H:%M")
        fname    = datetime.now().strftime("%Y%m%d_%H%M")
        md_lines = [f"# {st.session_state.active_session} — {ts_now}\nModel: {cur_info['label']}\n"]
        for role, content, ts in chat_history():
            md_lines.append(f"**{'User' if role=='user' else 'Assistant'}** _{ts}_\n\n{content}\n")

        md_out   = "\n---\n".join(md_lines)
        txt_out  = "\n".join(f"[{ts}] {r.upper()}: {c}" for r, c, ts in chat_history())
        json_out = json.dumps(
            [{"role": r, "content": c, "timestamp": ts} for r, c, ts in chat_history()],
            indent=2)

        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 Markdown",   md_out,   file_name=f"chat_{fname}.md",   mime="text/markdown",    use_container_width=True)
        c2.download_button("📃 Plain Text", txt_out,  file_name=f"chat_{fname}.txt",  mime="text/plain",       use_container_width=True)
        c3.download_button("🗂️ JSON",       json_out, file_name=f"chat_{fname}.json", mime="application/json", use_container_width=True)

st.caption("Built with Streamlit · LangChain · Hugging Face")