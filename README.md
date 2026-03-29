# Notebook to PDF Agent

This assignment builds a LangChain-based AI agent that converts a Jupyter Notebook (`.ipynb`) into a polished PDF report. The agent uses an Ollama LLM to orchestrate three specialized tools for parsing, formatting, and rendering.

## Architecture

The system uses a **LangChain agent** with the following tools:

1. **NotebookParserTool** - Parses `.ipynb` files and extracts all cells (markdown, code, outputs) as normalized JSON
2. **FormatterTool** - Formats parsed notebook content into render-ready blocks (markdown, code, output tuples)
3. **PDFGeneratorTool** - Generates a professional PDF from formatted content using ReportLab

The agent receives natural language instructions and autonomously calls these tools in the correct sequence.

## Features

- **LLM-powered orchestration**: ChatOllama (via LangChain) manages tool execution flow
- **Intelligent parsing**: Extracts and normalizes all cell types from Jupyter notebooks
- **Professional PDF output**: Renders with styled markdown (headings/lists), code blocks, output blocks, and spacing
- **Configurable LLM**: Model and temperature set via `.env` file environment variables
- **Flexible content handling**: Supports markdown cells, code cells with syntax preservation, and various output types

## Project Structure

- [nb2pdf_agent.py](nb2pdf_agent.py): End-to-end converter (parser, formatter, PDF generator)
- [main.py](main.py): Thin launcher entry point
- [notebook.ipynb](notebook.ipynb): Sample input notebook
- [output.pdf](output.pdf): Generated output
- [sample_output.pdf](sample_output.pdf): Deliverable demo PDF
- [pyproject.toml](pyproject.toml): Project metadata and dependencies

## Requirements

- Python `>= 3.11.14`
- Dependencies (installed via `pyproject.toml`):
  - `nbformat`: For parsing Jupyter notebooks
  - `markdown`: For markdown rendering
  - `reportlab`: For PDF generation
  - `langchain`: For agent framework and tools
  - `langchain-ollama`: For ChatOllama LLM integration
  - `python-dotenv`: For environment variable management
  - `ollama`: Ollama runtime (must be running locally)

## Installation

Install from [pyproject.toml](pyproject.toml):

```bash
uv sync
```

Ensure Ollama is running locally with the desired model available (default: `minimax-m2.5:cloud`).

## Configuration

Set model parameters in `.env` file:

```bash
MODEL=minimax-m2.5:cloud
TEMPERATURE=0.7
```

The agent's system prompt is: "You are a helpful assistant for converting Jupyter notebooks into PDF reports. Follow the instructions step by step. If user provides other files (not Jupyter notebooks .ipynb files) to convert simply respond with 'Unsupported file type.'"

## Usage


run the agent script directly:

```bash
uv run nb2pdf_agent.py
```

The script will:
1. Parse `notebook.ipynb`
2. Have the agent orchestrate the three tools to convert it
3. Generate `output.pdf` with styled markdown, code blocks, and outputs

## Notes

- Input notebook path is fixed to `notebook.ipynb` in the script. To convert a different notebook, update the `file_path` variable in [nb2pdf_agent.py](nb2pdf_agent.py#L276).
- The agent uses `langchain.agents.create_agent` to orchestrate tools—the LLM decides which tool to call and in what order.
- Ollama must be running locally; ensure the model specified in `.env` (or default `minimax-m2.5:cloud`) is available.
- PDF output is always written to `output.pdf` in the current directory.
