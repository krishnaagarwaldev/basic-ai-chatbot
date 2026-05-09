import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ──────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # "deepseek-ai/deepseek-llm-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
]

COMPARE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # "deepseek-ai/deepseek-llm-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
]

ASSISTANT_MODES = {
    "General":            "You are a helpful, concise AI assistant.",
    "Python Expert":      "You are a senior Python developer. Provide clean, well-commented code with explanations.",
    "Data Scientist":     "You are an expert in ML, Data Science, and GenAI. Use precise technical language with examples.",
    "Interviewer":        "You are a professional technical interviewer. Ask focused questions and evaluate answers critically.",
    "Career Coach":       "You help people with career guidance, resume tips, and professional communication.",
    "Research Assistant": "You provide thorough, well-sourced research summaries. Always mention confidence levels.",
    "Explain Like I'm 5": "Explain everything in the simplest possible way using analogies and examples a child would understand.",
    "Socratic Tutor":     "Instead of giving answers directly, guide the user to discover the answer themselves through questions.",
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
    defaults = {
        "chat_history":      [],   # list of (role, content, timestamp)
        "lc_history":        [],   # LangChain message objects
        "chat_count":        0,
        "last_user_input":   "",
        "current_mode":      None,
        "response_times":    [],   # track latency per response
        "search_provider":   "DuckDuckGo",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    model_name = st.selectbox("Model", MODELS)

    assistant_mode = st.selectbox("Assistant Mode", list(ASSISTANT_MODES))

    response_language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Hinglish", "Spanish", "French", "German", "Japanese"],
    )

    st.divider()

    temperature  = st.slider("Temperature",    0.0,  1.0,  0.5, 0.1)
    max_tokens   = st.slider("Max Tokens",     256, 2048, 1024, 128)
    memory_window = st.slider("Memory Window", 2,    20,   10,   2,
                              help="Number of past messages sent as context")

    st.divider()

    web_search_enabled = st.toggle("🌐 Web Search")

    if web_search_enabled:
        search_provider = st.selectbox(
            "Search Provider",
            ["DuckDuckGo", "Wikipedia", "Both"],
        )
        st.session_state.search_provider = search_provider
        show_search_debug = st.toggle("Show Search Debug", value=False)
    else:
        show_search_debug = False

    compare_models  = st.toggle("🆚 Compare Models")
    show_timestamps = st.toggle("🕐 Timestamps", value=True)
    show_stats      = st.toggle("📊 Show Response Stats", value=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        for k in ["chat_history", "lc_history", "response_times"]:
            st.session_state[k] = []
        st.session_state.chat_count      = 0
        st.session_state.last_user_input = ""
        st.session_state.current_mode    = None
        st.rerun()

    # ── Sidebar Analytics ──────────────────
    if st.session_state.response_times:
        st.divider()
        st.caption("⚡ Session Stats")
        avg_t = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.metric("Avg Response Time", f"{avg_t:.1f}s")
        st.metric("Total Exchanges", st.session_state.chat_count)

# ──────────────────────────────────────────
# Sync System Prompt on Mode Change
# ──────────────────────────────────────────
system_prompt = ASSISTANT_MODES[assistant_mode]
if response_language != "English":
    system_prompt += f" Always respond in {response_language}."

if st.session_state.current_mode != (assistant_mode, response_language):
    st.session_state.current_mode = (assistant_mode, response_language)
    sys_msg = SystemMessage(content=system_prompt)
    if st.session_state.lc_history and isinstance(st.session_state.lc_history[0], SystemMessage):
        st.session_state.lc_history[0] = sys_msg
    else:
        st.session_state.lc_history.insert(0, sys_msg)

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
# Web Search Providers
# ──────────────────────────────────────────
def search_duckduckgo(query: str, max_results: int = 6) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        if not results:
            return "No results found."
        snippets = [f"[{r['title']}]\n{r['body']}" for r in results if r.get("body")]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"

def search_wikipedia(query: str) -> str:
    try:
        import wikipedia

        # Search for closest page
        results = wikipedia.search(query)

        if not results:
            return "No Wikipedia results found."

        page_title = results[0]

        # Fetch summary
        summary = wikipedia.summary(page_title, sentences=5)

        return f"[Wikipedia: {page_title}]\n{summary}"

    except wikipedia.exceptions.DisambiguationError as e:
        return f"Too many matches. Try being specific.\nOptions: {e.options[:5]}"

    except wikipedia.exceptions.PageError:
        return "Wikipedia page not found."

    except Exception as e:
        return f"Wikipedia search error: {e}"

def web_search(query: str) -> str:
    provider = st.session_state.search_provider

    if provider == "Wikipedia":
        return search_wikipedia(query)

    elif provider == "Both":
        wiki = search_wikipedia(query)
        ddg = search_duckduckgo(query)

        return f"{wiki}\n\n{ddg}"

    return search_duckduckgo(query)


def build_augmented_prompt(query: str, results: str) -> str:
    return (
        "You have access to the following LIVE web search results. "
        "You MUST use these results to answer accurately. "
        "Do NOT say you lack access to current information.\n\n"
        f"=== WEB SEARCH RESULTS ===\n{results}\n=== END ===\n\n"
        f"User question: {query}\n\n"
        "Answer based strictly on the search results above:"
    )

# ──────────────────────────────────────────
# History Helpers
# ──────────────────────────────────────────
def add_message(role: str, content: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append((role, content, ts))
    cls = HumanMessage if role == "user" else AIMessage
    st.session_state.lc_history.append(cls(content=content))

def get_context() -> list:
    sys_msgs  = [m for m in st.session_state.lc_history if isinstance(m, SystemMessage)]
    conv_msgs = [m for m in st.session_state.lc_history if not isinstance(m, SystemMessage)]
    return sys_msgs + conv_msgs[-memory_window:]

# ──────────────────────────────────────────
# Page Header
# ──────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.caption(
    f"Mode: **{assistant_mode}** · Model: `{model_name.split('/')[-1]}` · "
    f"Lang: **{response_language}**"
)

# ──────────────────────────────────────────
# Suggested Prompts
# ──────────────────────────────────────────
cols = st.columns(len(SUGGESTIONS))
selected_prompt = None
for col, s in zip(cols, SUGGESTIONS):
    if col.button(s, use_container_width=True):
        selected_prompt = s

st.divider()

# ──────────────────────────────────────────
# Display Chat History
# ──────────────────────────────────────────
for role, content, ts in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)
        if show_timestamps:
            st.caption(ts)

# ──────────────────────────────────────────
# Input Resolution
# ──────────────────────────────────────────
user_input = st.chat_input("Type your message…")
if selected_prompt:
    user_input = selected_prompt

col_regen, col_summary = st.columns([1, 1])

with col_regen:
    if st.button("🔄 Regenerate", disabled=not st.session_state.last_user_input):
        # Drop last AI reply
        if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "assistant":
            st.session_state.chat_history.pop()
        if st.session_state.lc_history and isinstance(st.session_state.lc_history[-1], AIMessage):
            st.session_state.lc_history.pop()
        user_input = st.session_state.last_user_input

with col_summary:
    summarize = st.button(
        "📝 Summarize Chat",
        disabled=len(st.session_state.chat_history) < 4,
    )

# ──────────────────────────────────────────
# Summarize Entire Chat
# ──────────────────────────────────────────
if summarize:
    full_text = "\n".join(
        f"{role.upper()}: {content}"
        for role, content, _ in st.session_state.chat_history
    )
    summary_prompt = f"Summarize this conversation in 5 bullet points:\n\n{full_text}"
    model = load_model(model_name, temperature, max_tokens)
    with st.expander("📝 Conversation Summary", expanded=True):
        with st.spinner("Summarizing…"):
            resp = model.invoke([HumanMessage(content=summary_prompt)])
            st.markdown(resp.content)

# ──────────────────────────────────────────
# Main Chat Logic
# ──────────────────────────────────────────
if user_input:
    raw = user_input.strip()
    st.session_state.last_user_input = raw

    # ── Web Search ─────────────────────────
    search_results = None
    if web_search_enabled:
        provider_label = st.session_state.search_provider
        with st.spinner(f"🌐 Searching {provider_label}…"):
            search_results = web_search(raw)
        lc_input = build_augmented_prompt(raw, search_results)
    else:
        lc_input = raw

    # Store & display user message
    add_message("user", raw)
    if web_search_enabled:
        # Patch lc_history to use augmented prompt
        st.session_state.lc_history[-1] = HumanMessage(content=lc_input)

    with st.chat_message("user"):
        st.markdown(raw)

    # ── Debug Panel ────────────────────────
    if web_search_enabled and show_search_debug:
        with st.expander("🔍 Search Results Used", expanded=False):
            st.text(search_results)
        with st.expander("📤 Full Prompt Sent to Model", expanded=False):
            st.code(lc_input, language="text")

    # ── Compare Models ─────────────────────
    if compare_models:
        st.subheader("🆚 Model Comparison")
        compare_cols = st.columns(len(COMPARE_MODELS))
        ctx = get_context()
        for col, repo in zip(compare_cols, COMPARE_MODELS):
            with col:
                st.markdown(f"**{repo.split('/')[-1]}**")
                try:
                    m    = load_model(repo, temperature, max_tokens)
                    resp = m.invoke(ctx)
                    st.write(resp.content)
                except Exception as e:
                    st.error(str(e))

    # ── Normal Chat ────────────────────────
    else:
        model = load_model(model_name, temperature, max_tokens)
        ctx   = get_context()

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            t0 = time.time()

            try:
                for chunk in model.stream(ctx):
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
                st.caption(f"⏱ {elapsed}s · {words} words · ~{est_tokens} tokens")

        add_message("assistant", full_response)
        st.session_state.chat_count += 1

# ──────────────────────────────────────────
# Export (collapsed)
# ──────────────────────────────────────────
st.divider()
with st.expander("📥 Export Chat"):
    if not st.session_state.chat_history:
        st.info("No messages yet.")
    else:
        lines = [f"# Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for role, content, ts in st.session_state.chat_history:
            label = "**User**" if role == "user" else "**Assistant**"
            lines.append(f"{label} _{ts}_\n\n{content}\n")
        md_export = "\n---\n".join(lines)

        txt_export = "\n".join(
            f"[{ts}] {role.upper()}: {content}"
            for role, content, ts in st.session_state.chat_history
        )

        c1, c2 = st.columns(2)
        c1.download_button(
            "📄 Markdown",
            data=md_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        c2.download_button(
            "📃 Plain Text",
            data=txt_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ──────────────────────────────────────────
# Footer
# ──────────────────────────────────────────
st.caption("Built with Streamlit · LangChain · Hugging Face")