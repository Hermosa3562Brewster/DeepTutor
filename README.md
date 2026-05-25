# DeepTutor

> An AI-powered tutoring system for interactive document-based learning.

DeepTutor is a fork of [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor) — an intelligent tutoring system that enables deep, conversational learning from PDF documents and research papers using large language models.

## ✨ Features

- 📄 **Document Understanding** — Upload PDFs and ask questions about their content
- 🧠 **Deep Reasoning** — Powered by advanced LLMs for nuanced, accurate answers
- 💬 **Conversational Interface** — Multi-turn dialogue with context retention
- 🔍 **Retrieval-Augmented Generation (RAG)** — Precise answers grounded in your documents
- 📊 **Knowledge Graph** — Builds structured knowledge from document content
- 🌐 **Web UI** — Clean, responsive interface for seamless interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, recommended)
- An OpenAI-compatible API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DeepTutor.git
cd DeepTutor

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your API keys and configuration.

### Running Locally

```bash
python app.py
```

Open your browser at `http://localhost:7860`.

> **Personal note:** I've found it useful to set `GRADIO_SERVER_NAME=0.0.0.0` in `.env` when running
> inside a dev container or WSL2, so the UI is reachable from the host browser.
>
> Also setting `GRADIO_SERVER_PORT=7861` avoids conflicts if you already have another Gradio app
> running on the default port.
>
> If the knowledge graph step feels slow on first document load, try setting `MAX_ASYNC=4` in `.env`
> to limit concurrent LLM calls — helps avoid rate-limit errors on free-tier API keys.
>
> **Tip:** If you're using Ollama locally instead of OpenAI, set `OPENAI_API_BASE=http://localhost:11434/v1`
> and `OPENAI_API_KEY=ollama` in `.env`. I've been using `llama3.1:8b` for most testing and it
> works well enough for single-paper Q&A, though the knowledge graph quality drops noticeably
> compared to GPT-4o.
>
> **Note:** I've also had good results with `mistral:7b` via Ollama for shorter documents — it's
> noticeably faster than llama3.1:8b on my machine and the Q&A accuracy is comparable.
>
> **Note:** `qwen2.5:7b` via Ollama has been even better in my recent testing — noticeably stronger
> at following the structured output format the knowledge graph pipeline expects, which means fewer
> parse errors and more complete graphs. Worth trying if `mistral:7b` gives you malformed JSON in
> the graph step.

### Running with Docker

```bash
docker build -t deeptutor .
docker run -p 7860:7860 --env-file .env deeptutor
```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt
```

> **Note:** The original README had a typo here (`requirements` instead of `requirements-dev.txt`).
> Fixed for clarity — make sure `requirements-dev.txt` exists or adjust to match whatever dev
> deps file is actually present in the repo.
