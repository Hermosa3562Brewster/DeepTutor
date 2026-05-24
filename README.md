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

### Running with Docker

```bash
docker build -t deeptutor .
docker run -p 7860:7860 --env-file .env deeptutor
```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Lint
ruff check .
```

## 📁 Project Structure

```
DeepTutor/
├── app.py                  # Main application entry point
├── core/                   # Core logic (RAG pipeline, LLM clients)
├── ui/                     # Gradio / frontend components
├── utils/                  # Shared utilities
├── tests/                  # Test suite
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](.github/pull_request_template.md) and open an issue before submitting a pull request.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- Original project: [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)
- Built with [LightRAG](https://github.com/HKUDS/LightRAG), [Gradio](https://gradio.app/), and [LangChain](https://langchain.com/)
