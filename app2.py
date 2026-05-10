import time
import json
import base64
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ──────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "🦙 General Chat": {
        "meta-llama/Llama-3.1-8B-Instruct":  {"label": "Llama 3.1 8B",     "tags": ["chat","fast"],      "context": "128k", "desc": "Meta's best open chat model."},
        # "meta-llama/Llama-3.2-3B-Instruct":  {"label": "Llama 3.2 3B",     "tags": ["lightweight"],      "context": "128k", "desc": "Smallest Llama — fastest responses."},
        # "mistralai/Mistral-7B-Instruct-v0.3": {"label": "Mistral 7B",        "tags": ["chat","efficient"], "context": "32k",  "desc": "Mistral's efficient 7B model."},
        # "mistralai/Mixtral-8x7B-Instruct-v0.1":{"label":"Mixtral 8x7B (MoE)","tags":["powerful","MoE"],  "context": "32k",  "desc": "Much stronger MoE architecture."},
    },
    "🧠 Reasoning & Math": {
        "Qwen/Qwen2.5-7B-Instruct":   {"label": "Qwen 2.5 7B",   "tags": ["reasoning","math"],         "context": "128k", "desc": "Great at reasoning and multilingual."},
        "Qwen/Qwen2.5-72B-Instruct":  {"label": "Qwen 2.5 72B ⭐","tags": ["powerful","best"],          "context": "128k", "desc": "Near GPT-4 quality — best free model."},
        # "microsoft/Phi-3-mini-4k-instruct":  {"label": "Phi-3 Mini",  "tags": ["small","fast"],         "context": "4k",   "desc": "Small but capable reasoning model."},
        # "microsoft/Phi-3-medium-4k-instruct":{"label": "Phi-3 Medium","tags": ["reasoning","balanced"],  "context": "4k",   "desc": "Stronger Phi variant."},
    },
    "💻 Coding": {
        # "Qwen/Qwen2.5-Coder-7B-Instruct":  {"label": "Qwen Coder 7B",   "tags": ["code","fast"],    "context": "128k", "desc": "Strong code generation."},
        "Qwen/Qwen2.5-Coder-32B-Instruct": {"label": "Qwen Coder 32B ⭐","tags": ["code","best"],    "context": "128k", "desc": "Best free coding model on HF."},
        # "deepseek-ai/DeepSeek-Coder-V2-Instruct":{"label":"DeepSeek Coder V2","tags":["code","powerful"],"context":"128k","desc":"DeepSeek's dedicated code model."},
    },
    "🌍 Multilingual": {
        "Qwen/Qwen2.5-7B-Instruct":          {"label": "Qwen 2.5 7B", "tags": ["Hindi","100+ langs"], "context": "128k", "desc": "Best for Hindi/Hinglish."},
        # "mistralai/Mistral-7B-Instruct-v0.3": {"label": "Mistral 7B",  "tags": ["European"],          "context": "32k",  "desc": "Good for European languages."},
    },
}

ALL_MODELS: dict[str, dict] = {}
for _cat, _models in MODEL_REGISTRY.items():
    for _repo, _info in _models.items():
        ALL_MODELS[_repo] = {**_info, "category": _cat}

COMPARE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
]

ASSISTANT_MODES = {
    "General":            "You are a helpful, concise AI assistant.",
    "Python Expert":      "You are a senior Python developer. Provide clean, well-commented code with explanations.",
    "Data Scientist":     "You are an expert in ML, Data Science, and GenAI. Use precise technical language.",
    "Interviewer":        "You are a professional technical interviewer. Ask focused questions and evaluate answers critically.",
    "Career Coach":       "You help people with career guidance, resume tips, and professional communication.",
    "Research Assistant": "You provide thorough, well-sourced research summaries. Always mention confidence levels.",
    "Explain Like I'm 5": "Explain everything simply using analogies a child would understand.",
    "Socratic Tutor":     "Guide users to discover answers themselves through questions rather than giving direct answers.",
    "Custom":             "",   # filled by user input
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
        "sessions":          {"Session 1": {"history": [], "lc_history": [], "count": 0}},
        "active_session":    "Session 1",
        "response_times":    [],
        "last_user_input":   "",
        "current_mode":      None,
        "search_provider":   "DuckDuckGo",
        "active_model":      "meta-llama/Llama-3.1-8B-Instruct",
        "ratings":           {},   # msg_index -> 👍/👎
        "prompt_templates":  {     # built-in saved templates
            "Debug my code":     "Please debug the following code and explain each fix:\n\n",
            "Explain concept":   "Explain the concept of [TOPIC] with examples:\n\n",
            "Write unit tests":  "Write comprehensive unit tests for:\n\n",
            "Summarize text":    "Summarize the following in 5 bullet points:\n\n",
        },
        "file_context":      "",   # extracted file text
        "custom_prompt":     "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Active session shortcuts ───────────────
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

    # ── Multi-Session Manager ──────────────
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

    # ── Model Picker ───────────────────────
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
        </div>""", unsafe_allow_html=True,
    )

    st.divider()

    # ── Assistant Mode ─────────────────────
    assistant_mode = st.selectbox("Assistant Mode", list(ASSISTANT_MODES))
    if assistant_mode == "Custom":
        custom_input = st.text_area(
            "Write your system prompt",
            value=st.session_state.custom_prompt,
            height=100,
            placeholder="You are a helpful assistant that...",
        )
        st.session_state.custom_prompt = custom_input
        ASSISTANT_MODES["Custom"] = custom_input

    response_language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Hinglish", "Spanish", "French", "German", "Japanese"],
    )

    st.divider()

    temperature   = st.slider("Temperature",  0.0, 1.0, 0.5, 0.1)
    max_tokens    = st.slider("Max Tokens",   256, 2048, 1024, 128)
    memory_window = st.slider("Memory Window", 2, 20, 10, 2)

    st.divider()

    web_search_enabled = st.toggle("🌐 Web Search")
    if web_search_enabled:
        search_provider   = st.selectbox("Search Provider", ["DuckDuckGo", "Wikipedia", "Both"])
        st.session_state.search_provider = search_provider
        show_search_debug = st.toggle("Show Search Debug", value=False)
    else:
        show_search_debug = False

    compare_models_on = st.toggle("🆚 Compare Models")
    show_timestamps   = st.toggle("🕐 Timestamps", value=True)
    show_stats        = st.toggle("📊 Response Stats", value=True)
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
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temp,
    )
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
    if p == "Wikipedia":
        return search_wikipedia(query)
    if p == "Both":
        return search_wikipedia(query) + "\n\n" + search_duckduckgo(query)
    return search_duckduckgo(query)

def build_augmented_prompt(query: str, results: str) -> str:
    return (
        "You have access to LIVE web search results below. "
        "You MUST use them to answer. Do NOT claim you lack current info.\n\n"
        f"=== WEB SEARCH RESULTS ===\n{results}\n=== END ===\n\n"
        f"User question: {query}\n\nAnswer using the search results:"
    )

# ──────────────────────────────────────────
# File Reader
# ──────────────────────────────────────────
def extract_file_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt") or name.endswith(".md") or name.endswith(".py") or name.endswith(".csv"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                return "[Install PyPDF2 to read PDFs: pip install PyPDF2]"
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
    lc   = lc_history()
    sys_msgs  = [m for m in lc if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in lc if not isinstance(m, SystemMessage)]
    # inject file context as a system note if present
    if st.session_state.file_context:
        file_note = SystemMessage(
            content=f"The user has uploaded a file. Its content:\n\n{st.session_state.file_context[:3000]}\n\nUse it to answer questions."
        )
        return sys_msgs + [file_note] + conv_msgs[-memory_window:]
    return sys_msgs + conv_msgs[-memory_window:]

# ──────────────────────────────────────────
# Header
# ──────────────────────────────────────────
cur_info = ALL_MODELS[model_name]
st.title("🤖 AI Chatbot")
st.caption(
    f"**{cur_info['label']}** · Mode: {assistant_mode} · "
    f"Lang: {response_language} · Session: **{st.session_state.active_session}**"
)

# ──────────────────────────────────────────
# Tabs: Chat | File Upload | Prompt Templates
# ──────────────────────────────────────────
tab_chat, tab_file, tab_templates = st.tabs(["💬 Chat", "📁 File Upload", "📋 Prompt Templates"])

# ══════════════════════════════════════════
# TAB: File Upload
# ══════════════════════════════════════════
with tab_file:
    st.subheader("📁 Upload a File to Chat With")
    st.caption("Supported: .txt, .md, .py, .csv, .json, .pdf")

    uploaded = st.file_uploader(
        "Upload file", type=["txt","md","py","csv","json","pdf"], label_visibility="collapsed"
    )

    if uploaded:
        extracted = extract_file_text(uploaded)
        st.session_state.file_context = extracted
        st.success(f"✅ Loaded **{uploaded.name}** — {len(extracted)} characters")
        with st.expander("Preview file content"):
            st.code(extracted[:2000], language="text")

    if st.session_state.file_context:
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"📎 File active: {len(st.session_state.file_context)} chars injected into context")
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

    # Save new template
    with st.expander("➕ Save a new template"):
        t_name = st.text_input("Template name")
        t_body = st.text_area("Template text", height=100)
        if st.button("Save Template") and t_name and t_body:
            st.session_state.prompt_templates[t_name] = t_body
            st.success(f"Saved: {t_name}")

    # List existing templates
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

    # Suggested Prompts
    cols = st.columns(len(SUGGESTIONS))
    selected_prompt = None
    for col, s in zip(cols, SUGGESTIONS):
        if col.button(s, use_container_width=True):
            selected_prompt = s

    st.divider()

    # ── Chat History ───────────────────────
    for i, (role, content, ts) in enumerate(chat_history()):
        with st.chat_message(role):
            st.markdown(content)
            row = st.columns([6, 1, 1, 1])

            if show_timestamps:
                row[0].caption(ts)

            # Copy button (shows text in expander — clipboard API not available in Streamlit natively)
            if role == "assistant":
                with st.expander("📋 Copy", expanded=False):
                    st.code(content, language="markdown")

            # Ratings
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
    user_input = st.chat_input("Type your message…")

    if selected_prompt:
        user_input = selected_prompt
    if pending_template:
        user_input = pending_template

    # Action buttons
    col_regen, col_summary, col_auto_title = st.columns(3)

    with col_regen:
        if st.button("🔄 Regenerate", disabled=not st.session_state.last_user_input):
            h  = chat_history()
            lc = lc_history()
            if h and h[-1][0] == "assistant":
                h.pop()
            if lc and isinstance(lc[-1], AIMessage):
                lc.pop()
            user_input = st.session_state.last_user_input

    with col_summary:
        summarize = st.button("📝 Summarize Chat", disabled=len(chat_history()) < 4)

    with col_auto_title:
        auto_title = st.button("🏷️ Auto-Title Session", disabled=len(chat_history()) < 2)

    # ── Auto-Title ─────────────────────────
    if auto_title:
        full_text = " ".join(c for _, c, _ in chat_history()[:4])
        model_obj = load_model(model_name, temperature, max_tokens)
        with st.spinner("Generating title…"):
            resp = model_obj.invoke([HumanMessage(
                content=f"Generate a 4-5 word title for this conversation. Only output the title, nothing else:\n\n{full_text[:500]}"
            )])
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
                resp = model_obj.invoke([HumanMessage(content=f"Summarize in 5 bullets:\n\n{full_text}")])
                st.markdown(resp.content)

    # ── File Context Banner ─────────────────
    if st.session_state.file_context:
        st.info("📎 File context active — the model can see your uploaded file.")

    # ── Main Chat Logic ────────────────────
    if user_input:
        raw = user_input.strip()
        st.session_state.last_user_input = raw

        # Web Search
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
            ctx  = get_context()
            ccols = st.columns(len(COMPARE_MODELS))
            for col, repo in zip(ccols, COMPARE_MODELS):
                with col:
                    rinfo = ALL_MODELS.get(repo, {})
                    st.markdown(f"**{rinfo.get('label', repo.split('/')[-1])}**")
                    try:
                        m    = load_model(repo, temperature, max_tokens)
                        resp = m.invoke(ctx)
                        st.write(resp.content)
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
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"⚠️ Error: {e}"
                    placeholder.error(full_response)

                elapsed = round(time.time() - t0, 2)
                st.session_state.response_times.append(elapsed)

                if show_stats:
                    words      = len(full_response.split())
                    est_tokens = len(full_response) // 4
                    st.caption(f"⏱ {elapsed}s · {words} words · ~{est_tokens} tokens · `{cur_info['label']}`")

                with st.expander("📋 Copy response", expanded=False):
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
            md_export  = "\n---\n".join(lines)
            txt_export = "\n".join(f"[{ts}] {r.upper()}: {c}" for r, c, ts in chat_history())

            # JSON export with ratings
            json_export = json.dumps([
                {
                    "role": r, "content": c, "timestamp": ts,
                    "rating": st.session_state.ratings.get(
                        f"rating_{st.session_state.active_session}_{i}", None
                    )
                }
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

            # Ratings summary
            all_ratings = list(st.session_state.ratings.values())
            if all_ratings:
                st.divider()
                st.caption(
                    f"⭐ Ratings — 👍 {all_ratings.count('👍')}  ·  👎 {all_ratings.count('👎')}"
                )

st.caption("Built with Streamlit · LangChain · Hugging Face")