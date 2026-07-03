"""Provider factory abstraction.

The original codebase exposed the :func:`fin_ai.core.request.create_llm_client`
function directly.  While useful, that function required callers to know all of
the keyword arguments (e.g., ``proxy_port``, ``http_proxy_port``) and it was a
bit unwieldy when used from higher‑level services or agents.

This module introduces :class:`ProviderFactory`, a thin wrapper around the
existing logic that returns a ready‑to‑use :class:`LiteLLMClient`.  The factory is
registered with provider names via the *provider registry* defined in
``request.py``.  New providers can be added by extending
``_PROVIDER_CONFIGS`` and optionally providing a custom builder.

The public API is intentionally small:

``get_provider(provider, **kwargs)``
    Construct a :class:`LiteLLMClient` for ``provider`` using the supplied
    keyword arguments.  It forwards to ``create_llm_client`` internally.

Design note: The factory keeps the `ProviderConfig` objects private; callers
interact only through provider names.  This pattern makes unit testing
straightforward – a test can supply stub values for keys like ``api_key`` or
``proxy_port`` without touching environment variables.
"""

from __future__ import annotations

from typing import Any, Dict

from .request import create_llm_client, get_provider_config  # re‑exported for type safety


class ProviderFactory:
    """Construct LLM clients from provider names.

    Example usage::

        factory = ProviderFactory()
        client = factory.get_provider("github", api_key="...")
    """

    def get_provider(self, provider: str, **kwargs: Any):  # pragma: no cover – trivial
        """Return a :class:`LiteLLMClient` for *provider*.

        Parameters are forwarded verbatim to :func:`create_llm_client`.  The
        method validates that the provider name is known; otherwise it raises
        :class:`ValueError`.
        """

        # Validate before delegating – this mirrors the logic in request.py but
        # keeps error handling local for callers.
        get_provider_config(provider)  # will raise ValueError if unknown
        return create_llm_client(provider, **kwargs)
