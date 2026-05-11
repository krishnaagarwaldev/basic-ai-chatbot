# 🤖 AI Chatbot

A feature-rich, multi-model AI chatbot built with **Streamlit**, **LangChain**, and **Hugging Face** — supporting math rendering, web search, file uploads, multi-session chat, and more.

---

## ✨ Features

- **Multi-Model Support** — Switch between Llama, Qwen, and other Hugging Face models grouped by category (General, Reasoning, Coding, Multilingual)
- **Assistant Modes** — Pre-built system prompts for General, Python Expert, Data Scientist, Interviewer, Career Coach, Research Assistant, ELI5, Socratic Tutor, and Custom
- **Math Rendering** — LaTeX equations rendered via KaTeX (`$inline$` and `$$display$$`)
- **Web Search** — Live DuckDuckGo search augments responses with current information
- **File Upload** — Chat with `.txt`, `.md`, `.py`, `.csv`, `.json`, and `.pdf` files
- **Multi-Session** — Create, switch, rename, and delete independent chat sessions
- **Prompt Templates** — Save and reuse custom prompt templates
- **Streaming Responses** — Real-time token-by-token output with a live preview
- **Chain-of-Thought** — `<think>` / `<reasoning>` blocks rendered as collapsible panels
- **Callout Boxes** — Styled info, warning, success, error, and tip callouts
- **Message Ratings** — 👍 / 👎 feedback on assistant responses
- **Response Stats** — Time, word count, and token estimate per response
- **Chat Export** — Download conversations as Markdown, plain text, or JSON
- **Multilingual** — Respond in English, Hindi, Hinglish, Spanish, French, German, or Japanese

---

## 🗂️ Project Structure

```
.
├── app.py               # Main Streamlit application
├── .env                 # Environment variables (HF token)
├── requirements.txt     # Python dependencies
└── README.md
```

## 🧭 Usage Guide

### Sidebar Controls

| Setting                      | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| Chat Sessions                | Create, switch, or delete sessions               |
| Model                        | Pick a category and model                        |
| Assistant Mode               | Choose a persona or write a custom system prompt |
| Response Language            | Set the language for replies                     |
| Temperature                  | Controls creativity (0 = precise, 1 = creative)  |
| Max Tokens                   | Maximum response length                          |
| Memory Window                | How many past messages the model remembers       |
| Web Search                   | Augment answers with live DuckDuckGo results     |
| Timestamps / Stats / Ratings | Toggle UI elements                               |

### Chat Tab

- Use **suggestion buttons** at the top for quick prompts
- Click **🔄 Regenerate** to retry the last response
- Click **📝 Summarize Chat** for a 5-bullet summary
- Click **🏷️ Auto-Title Session** to name the session automatically
- **📋 Copy raw** expander shows the unrendered markdown for each message

### File Upload Tab

Upload a file and then ask questions about it in the chat. Supported formats: `.txt`, `.md`, `.py`, `.csv`, `.json`, `.pdf`.

### Prompt Templates Tab

Save frequently used prompts and apply them with one click. Built-in templates include Debug my code, Explain concept, Write unit tests, and Summarize text.

### Exporting Chats

At the bottom of the Chat tab, expand **📥 Export Chat** to download the full conversation as Markdown, plain text, or JSON (with ratings included).

---

## 🎨 Rendering Features

| Feature         | Syntax                         |
| --------------- | ------------------------------ |
| Inline math     | `$E = mc^2$`                 |
| Display math    | `$$\int_0^\infty f(x)\,dx$$` |
| Code block      | ` ```python ` ... ` ``` `  |
| Info callout    | `> **info**: message`        |
| Warning callout | `> **warning**: message`     |
| Tip callout     | `> **tip**: message`         |
| Reasoning block | `<think>...</think>`         |

---

## 🔒 Notes

- Model inference runs via the **Hugging Face Inference API** — no GPU required locally.
- Web search uses **DuckDuckGo** with no API key needed.
- All chat history is stored in **Streamlit session state** and resets on page refresh.
- PDF reading requires **PyPDF2** (`pip install PyPDF2`).

---

*Built with ❤️ using Streamlit · LangChain · Hugging Face*
