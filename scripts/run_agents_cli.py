#!/usr/bin/env python3
"""Interactive CLI for running FinAI agent workflows from the terminal.

This script replicates the *Agent Workflows* section of
``dashboard/financial_analyst_dashboard.py`` without the Streamlit UI:

- pick an agent profile (``Research_Publisher`` — the publish agent — by
  default) from ``fin_ai.agents.agent_library``,
- select a provider / chat model / embedding configuration using the same
  logic and defaults as the dashboard,
- choose a prompt template from ``fin_ai.agents.prompts_library.RESEARCH_ANALYSIS``
  via an interactive numbered menu, or type a custom prompt (template
  ``{placeholders}`` such as ``{asset}`` are discovered and filled in),
- run the agent with :func:`fin_ai.core.processor.run_agent_task` and publish
  the final report (HTML or PDF), exactly like the dashboard does.

Usage::

    python scripts/run_agents_cli.py                # fully interactive
    python scripts/run_agents_cli.py --agent Research_Publisher --format pdf
    python scripts/run_agents_cli.py --template "Single Asset Analysis" --asset NVDA

Any value that is not supplied as a flag is collected interactively.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from string import Formatter

# Make ``fin_ai`` importable when the script is run directly (e.g. without the
# package being installed in editable mode).  Harmless if already on ``sys.path``.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fin_ai.config.fin_ai import (
    DEFAULT_PROVIDER,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_GITHUB_MODEL,
    DEFAULT_GITHUB_EMBEDDING_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    GITHUB_BASE_URL,
    GITHUB_EMBEDDING_BASE_URL,
    DEEPSEEK_BASE_URL,
    OLLAMA_BASE_URL,
    LOG_DIR,
)
from fin_ai.logging_config import configure_logging
from fin_ai.agents.agent_library import library as AGENT_LIBRARY
from fin_ai.agents.prompts_library import RESEARCH_ANALYSIS
from fin_ai.core.processor import (
    build_agent_llm_config,
    fetch_models,
    run_agent_task,
)
from fin_ai.core.request import get_provider_config, known_providers
from fin_ai.core.tools import publish_research_html, publish_research_pdf

logger = logging.getLogger("run_agents_cli")


def _looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic filter for embedding-capable model identifiers."""
    mid = (model_id or "").strip().lower()
    return "embedding" in mid or "embed" in mid


def _strip_terminate_marker(text: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith("TERMINATE"):
        cleaned = cleaned[: -len("TERMINATE")].rstrip()
    return cleaned


def _partial_format(template: str, mapping: dict[str, str]) -> str:
    """Replace ``{name}`` placeholders only for keys present in *mapping*."""

    def _repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        return mapping.get(name, "{" + name + "}")

    return re.sub(r"\{(\w+)\}", _repl, template)


def _discover_placeholders(template: str) -> list[str]:
    """Return unique placeholder field names used by *template*."""
    formatter = Formatter()
    names = [fname for _, fname, _, _ in formatter.parse(template) if fname]
    return list(dict.fromkeys(names))
# ---------------------------------------------------------------------------
# Interactive input helpers (plain ``input()`` — no third-party deps)
# ---------------------------------------------------------------------------


def _ask(prompt: str, default: str = "") -> str:
    """Prompt for a single-line value; blank returns *default*."""
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else default


def _ask_password(prompt: str, default: str = "") -> str:
    """Prompt for a secret value without echoing it back.

    Falls back to a plain ``input()`` if ``getpass`` is unavailable.
    """
    if not default and sys.stdin and sys.stdin.isatty():
        try:
            import getpass

            return getpass.getpass(f"{prompt}: ").strip()
        except (ImportError, EOFError, OSError):
            pass
    return _ask(prompt, default)


def _read_multiline(title: str) -> str:
    """Collect a multi-line prompt.

    Type the prompt on one or more lines; finish with a line containing only
    ``END`` (or two consecutive blank lines).
    """
    print(title)
    print("(finish with a line containing only END, or two blank lines)")
    lines: list[str] = []
    blanks = 0
    while True:
        try:
            raw = input("> ")
        except EOFError:
            break
        line = raw.strip()
        if line.upper() == "END":
            break
        if not line:
            blanks += 1
            if blanks >= 2:
                break
            continue
        blanks = 0
        lines.append(raw)
    return "\n".join(lines).strip()


def _select_numbered(
    title: str,
    items: list[str],
    default_index: int = 0,
    extra_hint: str = "",
) -> str:
    """Present a numbered menu and return the selected item.

    The user may reply with the item number, an exact (case-insensitive) match
    of the item text, or press Enter to accept the default.
    """
    if not items:
        raise ValueError("Nothing to choose from")
    default_index = max(0, min(default_index, len(items) - 1))
    print(f"\n{title}")
    if extra_hint:
        print(extra_hint)
    for idx, item in enumerate(items, start=1):
        marker = " (default)" if idx - 1 == default_index else ""
        print(f"  {idx:>2}. {item}{marker}")
    while True:
        try:
            raw = input(
                f"Enter choice [1-{len(items)}] or exact name (default: {default_index + 1}): "
            ).strip()
        except EOFError:
            return items[default_index]
        if not raw:
            return items[default_index]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
            print(f"  Invalid number. Choose 1-{len(items)}.")
            continue
        lowered = raw.lower()
        matches = [it for it in items if it.lower() == lowered]
        if matches:
            return matches[0]
        print("  No exact match — enter a number or an exact item name.")


def _confirm(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    try:
        raw = input(f"{prompt} [{default_str}]: ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ("y", "yes")

# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------


def _provider_label_to_key() -> dict[str, str]:
    """Map human-readable provider labels to registry keys."""
    return {cfg.label: name for name, cfg in known_providers().items()}


def resolve_provider(args: argparse.Namespace) -> str:
    """Choose a provider key (``ollama``/``github``/``deepseek``/proxied...)."""
    default = (args.provider or DEFAULT_PROVIDER or "ollama").strip().lower()
    if default not in known_providers():
        default = "ollama"
    label_to_key = _provider_label_to_key()
    labels = list(label_to_key.keys())
    default_label = next((l for l, k in label_to_key.items() if k == default), labels[0])
    label = _select_numbered(
        "Select provider:",
        labels,
        default_index=labels.index(default_label) if default_label in labels else 0,
    )
    return label_to_key[label]


def resolve_model(
    args: argparse.Namespace,
    provider: str,
    api_base: str,
    api_key: str,
) -> str:
    """Choose a chat model for *provider*, fetching available models."""
    cfg = get_provider_config(provider)
    if provider == "github":
        default = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    elif provider == "proxied_github":
        default = os.getenv("GITHUB_MODEL", DEFAULT_GITHUB_MODEL)
    elif provider in ("deepseek", "proxied_deepseek"):
        default = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    else:
        default = os.getenv("OLLAMA_MODEL", DEFAULT_CHAT_MODEL)

    listing = provider.removeprefix("proxied_")
    try:
        fetched = [m.id for m in fetch_models(listing, api_key=api_key, base_url=api_base) if getattr(m, "id", "")]
    except Exception as exc:  # noqa: BLE001 - fall back to defaults
        logger.warning("Could not fetch models for %s: %s", provider, exc)
        fetched = []

    options = fetched or [default]
    if args.model and args.model in options:
        return args.model
    if not fetched and args.model:
        options = [args.model, *options]
    return _select_numbered(
        f"Select model for '{cfg.label}':",
        options,
        default_index=options.index(default) if default in options else 0,
        extra_hint="(only the default is shown if the model endpoint is unreachable)",
    )


def resolve_embedding_config(args: argparse.Namespace) -> dict[str, str]:
    """Choose the embedding provider, model, base URL and (GitHub) token."""
    default_provider = (
        (args.embed_provider or DEFAULT_EMBEDDINGS_PROVIDER or "ollama").strip().lower()
    )
    emb_labels = ["Local Ollama", "GitHub Models"]
    emb_label_to_key = {"Local Ollama": "ollama", "GitHub Models": "github"}
    if default_provider not in emb_label_to_key.values():
        default_provider = "ollama"
    provider_keys = list(emb_label_to_key.values())
    default_idx = provider_keys.index(default_provider)

    label = _select_numbered(
        "Select embedding provider:",
        emb_labels,
        default_index=default_idx,
    )
    provider = emb_label_to_key[label]

    if provider == "github":
        base = args.embed_base or os.getenv("GITHUB_EMBEDDING_BASE_URL", GITHUB_EMBEDDING_BASE_URL)
        base = _ask("GitHub embedding base URL", base)
        token = args.github_token or os.getenv("GITHUB_TOKEN", "")
        token = _ask_password("GitHub token (embeddings)", token)
        listing = "github"
        listing_key = token
        listing_base = base
        default_model = os.getenv("GITHUB_EMBEDDING_MODEL", DEFAULT_GITHUB_EMBEDDING_MODEL)
    else:  # ollama
        base = args.embed_base or os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL)
        base = _ask("Ollama endpoint (embeddings)", base)
        token = ""
        listing = "ollama"
        listing_key = ""
        listing_base = base
        default_model = os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    try:
        fetched = [
            m.id
            for m in fetch_models(listing, api_key=listing_key, base_url=listing_base)
            if getattr(m, "id", "") and _looks_like_embedding_model(m.id)
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch embedding models: %s", exc)
        fetched = []

    options = fetched or [default_model]
    if args.embed_model and args.embed_model in options:
        model = args.embed_model
    else:
        model = _select_numbered(
            f"Select embedding model for '{label}':",
            options,
            default_index=options.index(default_model) if default_model in options else 0,
        )
    return {
        "provider": provider,
        "model": model,
        "base_url": base,
        "github_token": token if provider == "github" else "",
    }


# ---------------------------------------------------------------------------
# Prompt template selection & building
# ---------------------------------------------------------------------------


def resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    """Choose a prompt template (or custom) and build the final task prompt.

    Returns ``(source_name, prompt_text)`` where ``source_name`` is either a
    template key from ``RESEARCH_ANALYSIS`` or ``"Custom"``.
    """
    template_keys = list(RESEARCH_ANALYSIS.keys())
    options = ["Custom", *template_keys]

    if args.template:
        if args.template == "Custom":
            source = "Custom"
        elif args.template in RESEARCH_ANALYSIS:
            source = args.template
        else:
            logger.warning("Unknown template %r — falling back to Custom.", args.template)
            source = "Custom"
    else:
        source = _select_numbered(
            "Select prompt template:",
            options,
            default_index=0,
            extra_hint="Pick a template from the library or choose 'Custom' to type your own.",
        )

    if source == "Custom":
        if args.prompt:
            prompt = args.prompt.strip()
        else:
            prompt = _read_multiline("Type your agent task prompt:")
    else:
        template_text = RESEARCH_ANALYSIS[source]
        placeholders = _discover_placeholders(template_text)

        params: dict[str, str] = {}
        if placeholders:
            print(f"\nTemplate '{source}' uses the following parameter(s): {', '.join(placeholders)}")
        for fname in placeholders:
            default_val = ""
            if fname == "asset" and args.asset:
                default_val = args.asset
            val = _ask(f"  Parameter: {fname}", default_val)
            if val:
                params[fname] = val.strip()

        if args.asset and args.asset.strip():
            params.setdefault("asset", args.asset.strip())

        prompt = _partial_format(template_text, params).strip()

    if not prompt:
        print("Prompt resolved to an empty string — nothing to run.")
        raise SystemExit(2)

    print("\n---- Resolved prompt ----")
    print(prompt)
    print("-------------------------")
    if _confirm("Use this prompt?", default=True):
        return source, prompt

    # Allow appending an extra instruction to the resolved prompt.
    extra = _read_multiline("Type extra instructions to append (or END to keep as-is):")
    if extra:
        prompt = f"{prompt}\n\n{extra}".strip()
    return source, prompt


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def run_agent(settings: dict, publish: bool = True) -> int:
    """Execute the agent task and publish the report; returns an exit code."""
    agent_name = settings["agent"]
    provider = settings["provider"]
    model = settings["model"]
    report_format = settings["format"]
    title = settings["title"].strip()
    email = settings["email"].strip()
    prompt = settings["prompt"].strip()

    print(f"\n== Running agent '{agent_name}' ==")
    print(f"   provider={provider} model={model}")
    print(f"   format={report_format} title={title or '(default)'}")
    if email:
        print(f"   email={email}")

    # Build the AutoGen-compatible llm_config for the chosen provider.
    llm_config = build_agent_llm_config(
        provider=provider,
        model=model,
        ollama_base_url=settings["chat_base"],
        github_endpoint=settings["github_endpoint"],
        github_token=settings["github_token"],
        deepseek_base_url=settings["deepseek_base_url"],
        deepseek_token=settings["deepseek_token"],
    )

    # Progress callback mirroring the dashboard's status bar.
    def _progress(pct: int, msg: str | None = None) -> None:
        msg = msg or ""
        print(f"  [{pct:>3}%] {msg}")

    result = run_agent_task(
        agent_name=agent_name,
        prompt=prompt,
        llm_config=llm_config,
        embedding_model=settings["embedding"]["model"],
        embedding_provider=settings["embedding"]["provider"],
        embedding_base_url=settings["embedding"]["base_url"],
        chat_provider=provider,
        is_publisher=(agent_name == "Research_Publisher"),
        publisher_format=report_format,
        publisher_email=email,
        publisher_title=title,
        progress_callback=_progress,
    )

    if not result["success"]:
        print(f"\nAgent error: {result.get('error')}")
        if result.get("trace"):
            print("\n--- trace ---")
            print(result["trace"])
        return 1

    response = result["response"]
    print("\n==================== AGENT RESPONSE ====================")
    print(_strip_terminate_marker(response))
    print("========================================================")

    # Publish the report in the chosen format — always, for any agent, exactly
    # like the dashboard does after a successful run. Skipped when ``--no-publish``.
    if not publish:
        print("\n(publication skipped: --no-publish)")
        return 0
    report_title = title or f"{agent_name} Report"
    content = _strip_terminate_marker(response) or prompt
    try:
        if report_format == "pdf":
            publication = publish_research_pdf(content, title=report_title)
        else:
            publication = publish_research_html(content, title=report_title)
    except Exception as exc:  # noqa: BLE001
        print(f"\nPublishing failed: {exc}")
        return 1

    try:
        pub_data = json.loads(publication) if isinstance(publication, str) else publication
        filepath = pub_data.get("filepath")
    except (json.JSONDecodeError, AttributeError):
        pub_data = None
        filepath = None

    print("\n---- Publication result ----")
    print(publication if isinstance(publication, str) else json.dumps(publication, indent=2))
    if filepath:
        print(f"\nPublished report: {filepath}")
    return 0

# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FinAI agent workflows from the terminal (dashboard-equivalent). "
            "Defaults to the publish agent (Research_Publisher). Any value not "
            "supplied as a flag is collected interactively."
        )
    )
    parser.add_argument("--agent", default="", help="Agent profile name (default: Research_Publisher)")
    parser.add_argument("--provider", default="", help="Provider key: ollama/github/deepseek/proxied_*")
    parser.add_argument("--model", default="", help="Chat model id")
    parser.add_argument("--embed-provider", default="", help="Embedding provider: ollama/github")
    parser.add_argument("--embed-model", default="", help="Embedding model id")
    parser.add_argument("--embed-base", default="", help="Embedding base URL")
    parser.add_argument("--github-token", default="", help="GitHub token (chat + embeddings if github)")
    parser.add_argument("--deepseek-token", default="", help="DeepSeek token")
    parser.add_argument("--api-base", default="", help="Chat provider base URL (overrides default)")
    parser.add_argument("--format", default="", choices=["", "html", "pdf"], help="Report format")
    parser.add_argument("--template", default="", help="Prompt template key from the prompts library")
    parser.add_argument("--prompt", default="", help="Custom prompt text (used with --template Custom)")
    parser.add_argument("--asset", default="", help="Asset symbol used for {asset} template params")
    parser.add_argument("--title", default="", help="Research report title")
    parser.add_argument("--email", default="", help="Email the report to this address")
    parser.add_argument("--no-publish", action="store_true", help="Skip the final publish step")
    return parser


def collect_settings(args: argparse.Namespace) -> dict:
    """Gather all settings interactively (flags become defaults)."""
    provider = resolve_provider(args)
    get_provider_config(provider)  # validates provider key

    # Provider-scoped tokens / base URLs.
    if provider in ("github", "proxied_github"):
        github_token = args.github_token or os.getenv("GITHUB_TOKEN", "")
        if provider == "github":
            github_token = _ask_password("GitHub token", github_token)
        base = args.api_base or os.getenv("GITHUB_ENDPOINT", GITHUB_BASE_URL)
        base = _ask("GitHub API base URL", base)
        deepseek_token = ""
        deepseek_base_url = DEEPSEEK_BASE_URL
        chat_base = base
        github_endpoint = base
    elif provider in ("deepseek", "proxied_deepseek"):
        deepseek_token = args.deepseek_token or os.getenv("DEEPSEEK_TOKEN", "")
        if provider == "deepseek":
            deepseek_token = _ask_password("DeepSeek token", deepseek_token)
        base = args.api_base or os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
        base = _ask("DeepSeek API base URL", base)
        github_token = ""
        github_endpoint = GITHUB_BASE_URL
        chat_base = base
        deepseek_base_url = base
    else:  # ollama
        base = args.api_base or os.getenv("OLLAMA_ENDPOINT", OLLAMA_BASE_URL)
        base = _ask("Ollama endpoint (chat)", base)
        github_token = ""
        github_endpoint = GITHUB_BASE_URL
        deepseek_token = ""
        deepseek_base_url = DEEPSEEK_BASE_URL
        chat_base = base

    model = resolve_model(args, provider, chat_base, github_token)

    # Embedding config.
    embedding = resolve_embedding_config(args)

    # Agent profile — default to the publish agent.
    agent_names = list(AGENT_LIBRARY.keys())
    default_agent = args.agent or "Research_Publisher"
    if default_agent not in agent_names:
        default_agent = "Research_Publisher"
    agent = _select_numbered(
        "Select agent profile:",
        agent_names,
        default_index=agent_names.index(default_agent) if default_agent in agent_names else 0,
    )

    if args.format in ("html", "pdf"):
        report_format = args.format
    else:
        report_format = _select_numbered("Select report format:", ["html", "pdf"], default_index=0)

    # Prompt template / custom prompt.
    source, prompt = resolve_prompt(args)
    if not prompt:
        print("No prompt provided — nothing to run.")
        raise SystemExit(2)

    title = args.title or _ask("Research name (report title, optional)", "")
    email = args.email or _ask("Email report to (optional)", "")

    return {
        "agent": agent,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "prompt_source": source,
        "format": report_format,
        "title": title,
        "email": email,
        "chat_base": chat_base,
        "github_endpoint": github_endpoint,
        "github_token": github_token,
        "deepseek_base_url": deepseek_base_url,
        "deepseek_token": deepseek_token,
        "embedding": embedding,
        "publish": not args.no_publish,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(log_dir=str(LOG_DIR))

    try:
        return _main_flow(args)
    except KeyboardInterrupt:
        # User pressed Ctrl-C (e.g. while an agent/Local LLM call is blocking).
        # Exit cleanly without a traceback.
        print("\n\nInterrupted by user (Ctrl-C). Aborting run.")
        return 130


def _main_flow(args: argparse.Namespace) -> int:
    if not (sys.stdin and sys.stdin.isatty()) and not args.prompt and not args.template:
        # Running non-interactively without enough information — refuse rather
        # than hang waiting for input.
        print(
            "Interactive prompts require a TTY. Supply --template/--prompt and "
            "other flags (see --help) for non-interactive runs.",
            file=sys.stderr,
        )
        return 2

    settings = collect_settings(args)

    print("\n============ RUN CONFIGURATION ============")
    print(f"  Agent            : {settings['agent']}")
    print(f"  Provider / Model : {settings['provider']} / {settings['model']}")
    print(f"  Chat base        : {settings['chat_base']}")
    print(f"  Embedding        : {settings['embedding']['provider']} / {settings['embedding']['model']}")
    print(f"  Format           : {settings['format']}")
    print(f"  Prompt source    : {settings['prompt_source']}")
    print("===========================================")

    if not _confirm("Run agent now?", default=True):
        print("Aborted.")
        return 1

    return run_agent(settings, publish=settings["publish"])


if __name__ == "__main__":
    raise SystemExit(main())
