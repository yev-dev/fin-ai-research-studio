# Project Architecture

## Overview
The project is split into two major packages:

| Package | Purpose |
|---------|---------|
| `fin_ai` | Domain‑specific AI/LLM components that augment the core library with RAG pipelines, prompt engineering and automated research publication. |

Both packages are laid out as conventional Python libraries following PEP 420 namespace package rules (no `__init__.py` in top level directories). The source code lives under `qf/` and `fin_ai/src/fin_ai/` respectively.

## Key Sub‑packages
- **qf.core** – foundational data structures (`DataFrame`, `Series`) wrappers, generic processors, and helper utilities.  It is used by the other risk modules.
- **qf.risk** – back‑testing engine, VaR calculations, corporate governance metrics and data loaders for historical price files.
- **qf.timeseries** – time‑series analytics such as clustering, gap filling, distressed‑stock detection and reproducibility helpers.
- **fin_ai.core** – LLM service wrappers, embeddings, prompt templates, rag pipeline orchestration, and utilities for publishing research.
- **fin_ai.agents** – high‑level agent workflows that combine the core services with external APIs (e.g., market data via `yfinance`).

## Architectural Decisions
1. **Isolation of AI logic** – All LLM‑related code lives under `fin_ai` so that the core financial engine can be used without requiring heavy dependencies like OpenAI or HuggingFace.
2. **Service factory pattern** – Each external service (e.g., `MarketDataService`) is instantiated via a static method `from_environment()` to read credentials from environment variables, making unit testing straightforward with mocks.
3. **JSON serialisation helpers** – A small set of helper functions (`_to_json_value`, `_dataframe_to_records`) are shared across the AI tools to convert pandas objects into JSON‑serialisable payloads.
4. **Documentation generation hooks** – Files that need to be processed by Cline are marked with `# cfn` directives (e.g., `# @cli: generate_docs`). This keeps the CI pipeline agnostic to where the docs live.
5. **Testing strategy** – Tests import from the top‑level packages (`qf`, `fin_ai`) and use isolated fixtures. Integration tests mock external services via `responses` or custom fixture `market_service_mock`.

## Future Work
- Add a thin wrapper around the back‑testing engine to expose an API similar to `backtrader` for ease of integration.
- Introduce a plugin system for custom risk models that can be loaded dynamically.
- Expand the RAG pipeline to support multi‑modal embeddings (image + text).

This architectural map serves as a reference for new contributors and guides future refactorings while keeping the codebase modular and testable.
