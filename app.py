import time
import json
import re
import io
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS + JS  (KaTeX · Mermaid · Copy buttons · Callouts)
# ══════════════════════════════════════════════════════════════
st.markdown(r"""
<!-- KaTeX -->
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{
    delimiters:[
      {left:'$$',right:'$$',display:true},
      {left:'$', right:'$', display:false},
      {left:'\\[',right:'\\]',display:true},
      {left:'\\(',right:'\\)',display:false}
    ],throwOnError:false});"></script>

<!-- Mermaid -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({startOnLoad:false, theme:'dark',
    themeVariables:{background:'#0d1117',primaryColor:'#1f6feb',
    primaryTextColor:'#e6edf3',lineColor:'#58a6ff',secondaryColor:'#161b22'}});
  function renderMermaid(){
    document.querySelectorAll('.mermaid-src').forEach(el=>{
      if(el.dataset.rendered) return;
      el.dataset.rendered='1';
      const div=document.createElement('div');
      div.className='mermaid';
      div.textContent=el.textContent;
      el.parentNode.replaceChild(div,el);
      mermaid.init(undefined,div);
    });
  }
  new MutationObserver(renderMermaid).observe(document.body,{childList:true,subtree:true});
</script>

<!-- Clipboard copy helper -->
<script>
function copyCode(id){
  const el=document.getElementById(id);
  if(!el) return;
  navigator.clipboard.writeText(el.innerText).then(()=>{
    const btn=document.querySelector('[data-copy="'+id+'"]');
    if(btn){btn.innerText='Copied!';setTimeout(()=>btn.innerText='Copy',2000);}
  });
}
</script>

<style>
/* KaTeX */
.katex{font-size:1.15rem!important}
.katex-display{overflow-x:auto;padding:.5rem 0}

/* Code wrapper */
.code-wrap{position:relative;margin:.8rem 0}
.code-wrap pre{
  background:#0d1117!important;border:1px solid #30363d!important;
  border-radius:8px!important;padding:2.6rem 1rem 1rem 1rem!important;
  overflow-x:auto!important;
  font-family:'JetBrains Mono','Fira Code','Cascadia Code',monospace!important;
  font-size:.88rem!important;line-height:1.65!important;margin:0}
.code-wrap code{
  background:transparent!important;border:none!important;
  padding:0!important;color:#e6edf3!important;font-size:inherit!important}
.code-toolbar{
  position:absolute;top:0;left:0;right:0;
  display:flex;align-items:center;justify-content:space-between;
  padding:5px 12px;border-bottom:1px solid #21262d;
  background:#161b22;border-radius:8px 8px 0 0;z-index:1}
.lang-badge{font-size:.72rem;font-family:monospace;text-transform:uppercase;
  font-weight:600;letter-spacing:.05em}
.copy-btn{
  font-size:.72rem;background:#21262d;color:#8b949e;border:1px solid #30363d;
  border-radius:4px;padding:2px 10px;cursor:pointer;transition:all .15s;
  font-family:inherit}
.copy-btn:hover{background:#388bfd22;color:#58a6ff;border-color:#388bfd}

/* Inline code */
code{
  font-family:'JetBrains Mono','Fira Code',monospace!important;
  font-size:.88em!important;background:#161b22!important;
  padding:2px 5px!important;border-radius:4px!important;
  border:1px solid #30363d!important;color:#79c0ff!important}

/* Tables */
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

/* Diff highlighting */
.diff-add{background:#1a4721;color:#3fb950;display:block;padding:0 4px;
  border-left:3px solid #3fb950;margin-left:-4px}
.diff-rem{background:#4a1e1e;color:#f85149;display:block;padding:0 4px;
  border-left:3px solid #f85149;margin-left:-4px}
.diff-info{background:#162535;color:#58a6ff;display:block;padding:0 4px;
  border-left:3px solid #58a6ff;margin-left:-4px}
.diff-ctx{color:#8b949e;display:block;padding:0 4px}

/* Callout boxes */
.callout{border-radius:8px;padding:.75rem 1rem;margin:.8rem 0;
  display:flex;gap:.6rem;align-items:flex-start;font-size:.93rem}
.callout-icon{font-size:1.1rem;flex-shrink:0;margin-top:1px}
.callout.info   {background:#131f2e;border:1px solid #1f6feb;color:#58a6ff}
.callout.warning{background:#2b1f0a;border:1px solid #9e6a03;color:#e3b341}
.callout.success{background:#0d2119;border:1px solid #238636;color:#3fb950}
.callout.error  {background:#2a0e0e;border:1px solid #da3633;color:#f85149}
.callout.tip    {background:#1a1f2e;border:1px solid #6e40c9;color:#d2a8ff}

/* Think / reasoning block */
details.think-block{
  background:#0d1117;border:1px solid #30363d;border-radius:8px;
  margin:.6rem 0;padding:0}
details.think-block summary{
  padding:.5rem .8rem;cursor:pointer;color:#8b949e;font-size:.82rem;
  font-style:italic;list-style:none;user-select:none}
details.think-block summary::-webkit-details-marker{display:none}
details.think-block summary::before{content:"🧠 ";margin-right:4px}
details.think-block .think-body{
  padding:.6rem 1rem .8rem;color:#8b949e;font-size:.88rem;
  border-top:1px solid #21262d;font-style:italic;line-height:1.6;white-space:pre-wrap}

/* Blockquote */
blockquote{border-left:4px solid #58a6ff!important;padding:.5rem 1rem!important;
  margin:.8rem 0!important;background:#161b22!important;border-radius:0 6px 6px 0!important;
  color:#8b949e!important;font-style:italic!important}

/* Headers */
h1,h2,h3,h4{margin-top:1.2rem!important;margin-bottom:.5rem!important;
  font-weight:700!important;border-bottom:1px solid #21262d;padding-bottom:.3rem}

/* Stat badges */
.stat-badge{display:inline-block;background:#161b22;border:1px solid #30363d;
  border-radius:20px;padding:2px 10px;font-size:.78rem;color:#8b949e;margin-right:6px}

/* Mermaid */
.mermaid{background:#0d1117;border:1px solid #30363d;border-radius:8px;
  padding:1rem;margin:.8rem 0;overflow-x:auto}

/* Chat polish */
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

COMPARE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

ASSISTANT_MODES = {
    "General": """You are a helpful, concise AI assistant.
Always format mathematical expressions using LaTeX:
- Inline math: $expression$  |  Block equations: $$expression$$
Format code in fenced blocks with language tag: ```python
Use markdown tables for comparisons.
Use **bold** for key terms, > blockquotes for notes/tips.
For diagrams/flowcharts use mermaid: ```mermaid
For warnings use: > ⚠️ Warning: text
For tips use: > 💡 Tip: text""",

    "Python Expert": """You are a senior Python developer.
Provide clean, production-quality, well-commented code.
Always use fenced code blocks: ```python
Format equations with LaTeX. Use tables for comparisons.
Use mermaid for architecture diagrams: ```mermaid
Highlight important notes with: > ⚠️ Warning: or > 💡 Tip:""",

    "Data Scientist": """You are an expert in ML, Data Science, AI and GenAI.
Use precise technical explanations.
ALWAYS render mathematical formulas using LaTeX:
- Use $$...$$ for display equations  |  $...$ for inline math
Format code with proper language tags. Use markdown tables.
Use mermaid for flowcharts and pipelines.""",

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
        "search_provider":  "DuckDuckGo",
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
    web_search_enabled = st.toggle("🌐 Web Search")
    if web_search_enabled:
        search_provider   = st.selectbox("Search Provider", ["DuckDuckGo", "Wikipedia", "Both"])
        st.session_state.search_provider = search_provider
        show_search_debug = st.toggle("Show Search Debug", value=False)
    else:
        show_search_debug = False

    compare_models_on = st.toggle("🆚 Compare Models")
    show_timestamps   = st.toggle("🕐 Timestamps",      value=True)
    show_stats        = st.toggle("📊 Response Stats",  value=True)
    show_ratings      = st.toggle("⭐ Message Ratings", value=True)

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
def search_duckduckgo(query: str, max_results: int = 6) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        if not results:
            return "No results found."
        return "\n\n".join(f"[{r['title']}]\n{r['body']}" for r in results if r.get("body"))
    except Exception as e:
        return f"DuckDuckGo error: {e}"

def search_wikipedia(query: str) -> str:
    try:
        import wikipedia
        pages = wikipedia.search(query)
        if not pages:
            return "No Wikipedia results."
        summary = wikipedia.summary(pages[0], sentences=5)
        return f"[Wikipedia: {pages[0]}]\n{summary}"
    except Exception as e:
        return f"Wikipedia error: {e}"

def web_search(query: str) -> str:
    p = st.session_state.search_provider
    if p == "Wikipedia": return search_wikipedia(query)
    if p == "Both":      return search_wikipedia(query) + "\n\n" + search_duckduckgo(query)
    return search_duckduckgo(query)

def build_augmented_prompt(query: str, results: str) -> str:
    return (
        "You have access to LIVE web search results below. "
        "You MUST use them to answer. Do NOT claim you lack current info.\n\n"
        f"=== WEB SEARCH RESULTS ===\n{results}\n=== END ===\n\n"
        f"User question: {query}\n\nAnswer using the search results:")


# ══════════════════════════════════════════════════════════════
# ★★★  RENDERING ENGINE  ★★★
# ══════════════════════════════════════════════════════════════

import html as _html

_COPY_CTR = 0

LANG_COLORS = {
    "python":"#3572A5","js":"#F7DF1E","javascript":"#F7DF1E",
    "ts":"#3178C6","typescript":"#3178C6","html":"#E34C26","css":"#563D7C",
    "sql":"#E38C00","bash":"#4EAA25","sh":"#4EAA25","shell":"#4EAA25",
    "json":"#8BC34A","yaml":"#CB171E","c":"#555555","cpp":"#F34B7D",
    "java":"#B07219","go":"#00ADD8","rust":"#DEA584","r":"#198CE7",
    "markdown":"#083FA1","md":"#083FA1","diff":"#F0C040","mermaid":"#FF6B6B",
}

CALLOUT_MAP = {
    "warning":("warning","⚠️"), "tip":("tip","💡"),
    "info":("info","ℹ️"),       "note":("info","📝"),
    "success":("success","✅"), "error":("error","❌"),
    "danger":("error","🚨"),
}
CALLOUT_EMOJI = {
    "⚠️":"warning","💡":"tip","ℹ️":"info","✅":"success","❌":"error","🚨":"error",
}

# ── 1. LATEX NORMALISER ───────────────────────────────────────
# Converts Qwen-style bare LaTeX to $…$ / $$…$$ delimiters

_RAW_DISPLAY_RE = re.compile(
    r'(?<!\$)'
    r'(\\begin\{(?:equation|align|gather|multline)\*?'
    r'\}[\s\S]*?\\end\{(?:equation|align|gather|multline)\*?\})',
    re.DOTALL,
)
_RAW_BOXED_RE  = re.compile(r'(?<!\$)(\\boxed\{[^}]*\})(?!\$)')
_RAW_FRAC_LINE = re.compile(
    r'^(\s*)(\\frac\{[^}]*\}\{[^}]*\}'
    r'(?:\s*[=+\-]?\s*\\frac\{[^}]*\}\{[^}]*\})*)\s*$',
    re.MULTILINE,
)
_BRACKET_DISPLAY_RE = re.compile(r'\\\[([\s\S]*?)\\\]')
_PAREN_INLINE_RE    = re.compile(r'\\\((.*?)\\\)')


def _normalise_latex(text: str) -> str:
    """Wrap bare LaTeX constructs that Qwen emits without $…$ delimiters."""
    text = _BRACKET_DISPLAY_RE.sub(lambda m: f'$$\n{m.group(1).strip()}\n$$', text)
    text = _PAREN_INLINE_RE.sub(lambda m: f'${m.group(1).strip()}$', text)
    text = _RAW_DISPLAY_RE.sub(lambda m: f'$$\n{m.group(1)}\n$$', text)
    text = _RAW_BOXED_RE.sub(lambda m: f'$${m.group(1)}$$', text)
    text = _RAW_FRAC_LINE.sub(lambda m: f'{m.group(1)}$$\n{m.group(2)}\n$$', text)
    return text


# ── 2. THINK-BLOCK EXTRACTOR ──────────────────────────────────
# Handles fully-closed AND unclosed/mid-stream <think> tags

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

    # Handle unclosed opening tag (streaming artefact from Qwen)
    m = _THINK_OPEN_RE.search(text)
    if m:
        blocks.append((m.group(1).capitalize(), m.group(2).strip()))
        text = text[: m.start()].strip()

    return text, blocks


# ── 3. SYNTAX COLOURISERS ─────────────────────────────────────

def _colorize(code: str, lang: str) -> str:
    c = _html.escape(code)
    L = lang.lower()

    if L in ("python", ""):
        c = re.sub(
            r'\b(def|class|return|import|from|if|elif|else|for|while|'
            r'try|except|finally|with|as|pass|break|continue|lambda|'
            r'yield|raise|assert|del|global|nonlocal|and|or|not|in|is|'
            r'True|False|None|async|await)\b',
            r'<span style="color:#ff7b72">\1</span>', c)
        c = re.sub(
            r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
            r'<span style="color:#a5d6ff">\1</span>', c)
        c = re.sub(r'(#[^\n]*)', r'<span style="color:#8b949e">\1</span>', c)
        c = re.sub(r'\b(\d+\.?\d*)\b', r'<span style="color:#79c0ff">\1</span>', c)
        c = re.sub(
            r'\b(print|len|range|type|str|int|float|list|dict|set|tuple|'
            r'bool|open|zip|map|filter|enumerate|isinstance|hasattr|getattr|'
            r'setattr|super|property|staticmethod|classmethod|self|cls)\b',
            r'<span style="color:#d2a8ff">\1</span>', c)
        c = re.sub(r'(@\w+)', r'<span style="color:#ffa657">\1</span>', c)

    elif L in ("js","javascript","ts","typescript"):
        c = re.sub(
            r'\b(const|let|var|function|return|if|else|for|while|class|'
            r'import|export|from|async|await|try|catch|finally|new|this|'
            r'typeof|instanceof|true|false|null|undefined|of|in|default|'
            r'switch|case|break|continue|throw)\b',
            r'<span style="color:#ff7b72">\1</span>', c)
        c = re.sub(
            r'(`[^`]*`|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
            r'<span style="color:#a5d6ff">\1</span>', c)
        c = re.sub(r'(//[^\n]*|/\*[\s\S]*?\*/)', r'<span style="color:#8b949e">\1</span>', c)
        c = re.sub(r'\b(\d+\.?\d*)\b', r'<span style="color:#79c0ff">\1</span>', c)

    elif L == "sql":
        c = re.sub(
            r'\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP|BY|'
            r'ORDER|HAVING|INSERT|UPDATE|DELETE|CREATE|TABLE|INDEX|DROP|ALTER|'
            r'AND|OR|NOT|IN|IS|NULL|AS|DISTINCT|LIMIT|OFFSET|UNION|ALL|'
            r'COUNT|SUM|AVG|MAX|MIN|WITH|RETURNING)\b',
            r'<span style="color:#ff7b72">\1</span>', c, flags=re.IGNORECASE)
        c = re.sub(r"('[^']*')", r'<span style="color:#a5d6ff">\1</span>', c)
        c = re.sub(r'(--[^\n]*)', r'<span style="color:#8b949e">\1</span>', c)

    elif L in ("bash","sh","shell"):
        c = re.sub(
            r'\b(echo|cd|ls|mkdir|rm|cp|mv|grep|awk|sed|cat|chmod|'
            r'export|source|pip|python|python3|git|curl|wget|sudo|apt|'
            r'brew|npm|yarn|docker|kubectl|ssh|scp|tar|unzip)\b',
            r'<span style="color:#d2a8ff">\1</span>', c)
        c = re.sub(r'(#[^\n]*)', r'<span style="color:#8b949e">\1</span>', c)
        c = re.sub(r'(\$\w+|\$\{[^}]+\})', r'<span style="color:#79c0ff">\1</span>', c)
        c = re.sub(
            r'("([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\')',
            r'<span style="color:#a5d6ff">\1</span>', c)

    elif L == "json":
        c = re.sub(r'"([^"]+)"\s*:', r'<span style="color:#79c0ff">"\1"</span>:', c)
        c = re.sub(r':\s*"([^"]*)"', r': <span style="color:#a5d6ff">"\1"</span>', c)
        c = re.sub(r'\b(true|false|null)\b', r'<span style="color:#ff7b72">\1</span>', c)
        c = re.sub(r'\b(\d+\.?\d*)\b', r'<span style="color:#79c0ff">\1</span>', c)

    elif L in ("html", "xml"):
        c = re.sub(r'(&lt;\/?)([\w-]+)', r'\1<span style="color:#7ee787">\2</span>', c)
        c = re.sub(r'([\w-]+)(=)', r'<span style="color:#79c0ff">\1</span>\2', c)
        c = re.sub(r'(&quot;[^&]*&quot;)', r'<span style="color:#a5d6ff">\1</span>', c)
        c = re.sub(r'(&lt;!--[\s\S]*?--&gt;)', r'<span style="color:#8b949e">\1</span>', c)

    return c


def _diff_colorize(code: str) -> str:
    lines = _html.escape(code).split("\n")
    out = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f'<span class="diff-add">{line}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f'<span class="diff-rem">{line}</span>')
        elif line.startswith("@@"):
            out.append(f'<span class="diff-info">{line}</span>')
        else:
            out.append(f'<span class="diff-ctx">{line}</span>')
    return "\n".join(out)


# ── 4. CODE BLOCK HTML ────────────────────────────────────────

def _code_block_html(code: str, lang: str) -> str:
    global _COPY_CTR
    _COPY_CTR += 1
    cid   = f"cb_{_COPY_CTR}"
    color = LANG_COLORS.get(lang.lower(), "#8b949e")
    badge = lang.upper() if lang else "CODE"
    highlighted = _diff_colorize(code) if lang.lower() == "diff" else _colorize(code, lang)
    return (
        f'<div class="code-wrap">'
        f'<div class="code-toolbar">'
        f'<span class="lang-badge" style="color:{color}">{badge}</span>'
        f'<button class="copy-btn" data-copy="{cid}" onclick="copyCode(\'{cid}\')">'
        f'Copy</button></div>'
        f'<pre><code id="{cid}">{highlighted}</code></pre>'
        f'</div>')


def _mermaid_html(code: str) -> str:
    escaped = _html.escape(code)
    return f'<pre class="mermaid-src" style="display:none">{escaped}</pre>'


# ── 5. CALLOUT DETECTOR ───────────────────────────────────────

_CALLOUT_LINE_RE = re.compile(
    r'^>\s*'
    r'(?:(⚠️|💡|ℹ️|✅|❌|🚨)\s*)?'
    r'(?:\*\*)?(?:(warning|tip|info|note|success|error|danger))(?:\*\*)?\s*[:\-]?\s*(.*)',
    re.IGNORECASE)

def _try_callout(line: str):
    m = _CALLOUT_LINE_RE.match(line)
    if not m:
        return None
    emoji_raw = (m.group(1) or "").strip()
    label_raw = (m.group(2) or "").strip().lower()
    body      = (m.group(3) or "").strip()
    key = label_raw or CALLOUT_EMOJI.get(emoji_raw, "")
    cls, icon = CALLOUT_MAP.get(key, ("info", "ℹ️"))
    return (f'<div class="callout {cls}">'
            f'<span class="callout-icon">{icon}</span>'
            f'<span>{_html.escape(body)}</span></div>')


# ── 6. JSON/CSV → dataframe ───────────────────────────────────

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


# ── 7. MARKDOWN TABLE → HTML ──────────────────────────────────

def _md_tables_to_html(text: str) -> str:
    lines, output, i = text.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header_cells = [c.strip() for c in table_lines[0].split("|") if c.strip()]
                html = ("<table><thead><tr>"
                        + "".join(f"<th>{c}</th>" for c in header_cells)
                        + "</tr></thead><tbody>")
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.split("|") if c.strip()]
                    if cells:
                        html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                html += "</tbody></table>"
                output.append(html)
            else:
                output.extend(table_lines)
            continue
        output.append(line)
        i += 1
    return "\n".join(output)


# ── 8. MASTER SPLITTER ────────────────────────────────────────
# Fixed: non-greedy $$…$$ + handles inline $…$ (single-line only)

_SPLIT_RE = re.compile(
    r'(```[\w]*\n?[\s\S]*?```'   # fenced code block  (highest priority)
    r'|\$\$[\s\S]*?\$\$'         # display math block
    # r'|\$[^\$\n]+?\$'            # inline math (single line)
    r')',
    re.DOTALL,
)


# ── 9. KaTeX RE-RENDER SCRIPT ────────────────────────────────
# Injected after each streaming chunk so math updates in real-time

# _RERENDER_JS = (
#     '<script>(function(){'
#     'if(window.renderMathInElement){'
#     'renderMathInElement(document.body,{'
#     'delimiters:['
#     '{left:"$$",right:"$$",display:true},'
#     '{left:"$",right:"$",display:false},'
#     '{left:"\\\\[",right:"\\\\]",display:true},'
#     '{left:"\\\\(",right:"\\\\)",display:false}'
#     '],throwOnError:false});'
#     '}})()</script>'
# )


# ── 10. MASTER RENDERER ───────────────────────────────────────

def render_response(text: str):
    """Full professional render — Qwen-compatible."""

    # Step 0: normalise Qwen bare LaTeX → $…$ / $$…$$
    text = _normalise_latex(text)

    # Step 1: pull out <think> / <reasoning> blocks
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

    # Step 2: split on fenced blocks / display math / inline math
    parts = _SPLIT_RE.split(text)

    for part in parts:
        if not part.strip():
            continue

        # ── Fenced code block ──────────────────────────────────
        if part.startswith("```"):
            m = re.match(r'```(\w*)\n?([\s\S]*?)```', part, re.DOTALL)
            if not m:
                st.code(part)
                continue
            lang = m.group(1).strip()
            code = m.group(2)

            if lang.lower() == "mermaid":
                st.markdown(_mermaid_html(code), unsafe_allow_html=True)
                continue

            if lang.lower() in ("json", "csv") and _try_render_data(code, lang):
                continue

            st.markdown(_code_block_html(code, lang), unsafe_allow_html=True)

        # ── Display math  $$ … $$ ──────────────────────────────
        elif part.startswith("$$") and part.endswith("$$"):
            formula = part[2:-2].strip()
            if formula:
                try:
                    st.latex(formula)
                except Exception:
                    # Fallback: pass through to KaTeX auto-render
                    st.markdown(
                        f'<div style="text-align:center;padding:.5rem 0">{part}</div>',
                        unsafe_allow_html=True)

        # ── Inline math  $ … $ ────────────────────────────────
        # elif (part.startswith("$") and part.endswith("$")
        #       and not part.startswith("$$") and "\n" not in part):
        #     # Let KaTeX auto-render handle it inside a prose span
        #     st.markdown(
        #         f'<span style="line-height:1.75">{part}</span>',
        #         unsafe_allow_html=True)

        # ── Prose / markdown ───────────────────────────────────
        else:
            lines, plain_batch = part.split("\n"), []

            def _flush():
                nonlocal plain_batch
                if plain_batch:
                    chunk = _md_tables_to_html("\n".join(plain_batch))
                    st.markdown(
                        f'<div style="line-height:1.75">{chunk}</div>',
                        unsafe_allow_html=True)
                    plain_batch = []

            for line in lines:
                callout = _try_callout(line)
                if callout:
                    _flush()
                    st.markdown(callout, unsafe_allow_html=True)
                else:
                    plain_batch.append(line)
            _flush()


# ── 11. STREAMING RENDER ──────────────────────────────────────

def render_streaming_chunk(text: str, placeholder):
    """
    Lightweight streaming render with Qwen LaTeX normalisation.
    Triggers KaTeX re-render after each chunk.
    """
    normalised = _normalise_latex(text)
    # Strip unclosed <think> tags so they don't bleed into visible output
    normalised = re.sub(
        r'<(think|thinking|reasoning|scratchpad)>[\s\S]*$',
        '', normalised, flags=re.IGNORECASE
    ).strip()
    # placeholder.markdown(normalised + " ▌" + _RERENDER_JS, unsafe_allow_html=True)
    placeholder.markdown(
        f'<div class="latex-stream">{normalised} ▌</div>',
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────
# File Reader
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
    lc        = lc_history()
    sys_msgs  = [m for m in lc if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in lc if not isinstance(m, SystemMessage)]
    if st.session_state.file_context:
        file_note = SystemMessage(
            content=f"The user has uploaded a file. Its content:\n\n"
                    f"{st.session_state.file_context[:3000]}\n\nUse it to answer questions.")
        return sys_msgs + [file_note] + conv_msgs[-memory_window:]
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
tab_chat, tab_file, tab_templates = st.tabs(["💬 Chat", "📁 File Upload", "📋 Prompt Templates"])

# ══════════════════════════════════════════
# TAB: File Upload
# ══════════════════════════════════════════
with tab_file:
    st.subheader("📁 Upload a File to Chat With")
    st.caption("Supported: .txt, .md, .py, .csv, .json, .pdf")
    uploaded = st.file_uploader(
        "Upload file", type=["txt","md","py","csv","json","pdf"],
        label_visibility="collapsed")
    if uploaded:
        extracted = extract_file_text(uploaded)
        st.session_state.file_context = extracted
        st.success(f"✅ Loaded **{uploaded.name}** — {len(extracted)} characters")
        with st.expander("Preview file content"):
            st.code(extracted[:2000], language="text")
    if st.session_state.file_context:
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"📎 File active: {len(st.session_state.file_context)} chars in context")
        with col_b:
            if st.button("❌ Remove File"):
                st.session_state.file_context = ""
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

    # ── Auto-Title ─────────────────────────
    if auto_title:
        full_text = " ".join(c for _, c, _ in chat_history()[:4])
        model_obj = load_model(model_name, temperature, max_tokens)
        with st.spinner("Generating title…"):
            resp = model_obj.invoke([HumanMessage(
                content=f"Generate a 4-5 word title for this conversation. "
                        f"Only output the title, nothing else:\n\n{full_text[:500]}")])
            new_title = resp.content.strip().strip('"').strip("'")[:40]
            old = st.session_state.active_session
            st.session_state.sessions[new_title] = st.session_state.sessions.pop(old)
            st.session_state.active_session = new_title
            st.rerun()

    # ── Summarize ──────────────────────────
    if summarize:
        full_text = "\n".join(f"{r.upper()}: {c}" for r, c, _ in chat_history())
        model_obj = load_model(model_name, temperature, max_tokens)
        with st.expander("📝 Conversation Summary", expanded=True):
            with st.spinner("Summarizing…"):
                resp = model_obj.invoke([HumanMessage(
                    content=f"Summarize in 5 bullets:\n\n{full_text}")])
                render_response(resp.content)

    if st.session_state.file_context:
        st.info("📎 File context active — the model can see your uploaded file.")

    # ── Main Chat Logic ────────────────────
    if user_input:
        raw = user_input.strip()
        st.session_state.last_user_input = raw

        search_results = None
        if web_search_enabled:
            with st.spinner(f"🌐 Searching {st.session_state.search_provider}…"):
                search_results = web_search(raw)
            lc_input = build_augmented_prompt(raw, search_results)
        else:
            lc_input = raw

        add_message("user", raw)
        if web_search_enabled:
            lc_history()[-1] = HumanMessage(content=lc_input)

        with st.chat_message("user"):
            st.markdown(raw)

        if web_search_enabled and show_search_debug:
            with st.expander("🔍 Search Results", expanded=False):
                st.text(search_results)

        # Compare Models
        if compare_models_on:
            st.subheader("🆚 Model Comparison")
            ctx   = get_context()
            ccols = st.columns(len(COMPARE_MODELS))
            for col, repo in zip(ccols, COMPARE_MODELS):
                with col:
                    rinfo = ALL_MODELS.get(repo, {})
                    st.markdown(f"**{rinfo.get('label', repo.split('/')[-1])}**")
                    try:
                        m    = load_model(repo, temperature, max_tokens)
                        resp = m.invoke(ctx)
                        render_response(resp.content)
                    except Exception as e:
                        st.error(str(e))

        # Normal Chat
        else:
            model_obj = load_model(model_name, temperature, max_tokens)
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

                    # Final professional render
                    placeholder.empty()
                    render_response(full_response)

                except Exception as e:
                    full_response = f"⚠️ Error: {e}"
                    placeholder.error(full_response)

                elapsed = round(time.time() - t0, 2)
                st.session_state.response_times.append(elapsed)

                if show_stats:
                    words      = len(full_response.split())
                    est_tokens = len(full_response) // 4
                    st.markdown(
                        f'<span class="stat-badge">⏱ {elapsed}s</span>'
                        f'<span class="stat-badge">📝 {words} words</span>'
                        f'<span class="stat-badge">🔢 ~{est_tokens} tokens</span>'
                        f'<span class="stat-badge">🤖 {cur_info["label"]}</span>',
                        unsafe_allow_html=True)

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

st.caption("Built with Streamlit · LangChain · Hugging Face")