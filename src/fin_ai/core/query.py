from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence

from langchain_core.documents import Document

from .request import count_message_tokens, get_model_safe_input_budget, resolve_model_name, trim_text_to_tokens
from .rag import filter_documents_by_metadata

if TYPE_CHECKING:
    from .response import ModelResponse, Provider
else:
    ModelResponse = Any
    Provider = str


QueryMode = Literal["ensemble", "separate", "routed"]
SourceGrouping = Literal["vector_db", "filename", "source_type", "source"]


class RetrieverLike(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


@dataclass(slots=True)
class VectorStoreFilteredRetriever:
    vector_store: Any
    search_type: str = "mmr"
    search_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata_filter: dict[str, Any] = field(default_factory=dict)

    def invoke(self, query: str) -> list[Document]:
        search_kwargs = dict(self.search_kwargs)
        if self.metadata_filter:
            existing_filter = search_kwargs.get("filter") or {}
            search_kwargs["filter"] = {**existing_filter, **self.metadata_filter}
        retriever = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )
        return list(retriever.invoke(query))


@dataclass(slots=True)
class SourceRetrieverConfig:
    name: str
    retriever: RetrieverLike | Any
    description: str = ""
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class SourceRetrievalResult:
    source_name: str
    documents: list[Document]
    metadata_filter: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryRoutingDecision:
    selected_sources: list[str]
    reasoning: str = ""
    planner_used: bool = False


@dataclass(slots=True)
class MultiSourceQueryResult:
    query: str
    mode: QueryMode
    selected_sources: list[str]
    source_results: list[SourceRetrievalResult]
    combined_documents: list[Document]
    routing_decision: QueryRoutingDecision | None = None


@dataclass(slots=True)
class MultiSourcePromptResult:
    retrieval: MultiSourceQueryResult
    prompt: str
    source_prompts: dict[str, str]
    response: ModelResponse | None = None


@dataclass(slots=True)
class SourceCitation:
    source_name: str
    filenames: list[str]
    source_types: list[str]
    chunk_indices: list[int]
    section_headers: list[str]
    document_count: int


OVERSIZED_PROMPT_SUMMARY_SYSTEM_PROMPT = (
    "You compress retrieved financial evidence into concise factual notes. "
    "Preserve figures, dates, comparisons, caveats, and explicit uncertainties. "
    "Do not add new facts."
)


def build_source_retriever_configs(
    vector_store: Any,
    *,
    base_name: str,
    group_by: SourceGrouping = "vector_db",
    search_type: str = "mmr",
    search_k: int = 5,
    weight: float = 1.0,
) -> list[SourceRetrieverConfig]:
    """Create source configs from a vector store, optionally grouped by metadata."""
    if group_by == "vector_db":
        return [
            SourceRetrieverConfig(
                name=base_name,
                retriever=VectorStoreFilteredRetriever(
                    vector_store=vector_store,
                    search_type=search_type,
                    search_kwargs={"k": search_k},
                ),
                description=f"Vector database {base_name}",
                weight=weight,
                tags=(base_name,),
            )
        ]

    metadata_groups = _discover_metadata_groups(vector_store, group_by)
    if not metadata_groups:
        return [
            SourceRetrieverConfig(
                name=base_name,
                retriever=VectorStoreFilteredRetriever(
                    vector_store=vector_store,
                    search_type=search_type,
                    search_kwargs={"k": search_k},
                ),
                description=f"Vector database {base_name}",
                weight=weight,
                tags=(base_name,),
            )
        ]

    configs: list[SourceRetrieverConfig] = []
    for group_value in metadata_groups:
        filter_dict = {group_by: group_value}
        group_name = f"{base_name}:{group_value}"
        configs.append(
            SourceRetrieverConfig(
                name=group_name,
                retriever=VectorStoreFilteredRetriever(
                    vector_store=vector_store,
                    search_type=search_type,
                    search_kwargs={"k": search_k},
                    metadata_filter=filter_dict,
                ),
                description=f"{base_name} grouped by {group_by}={group_value}",
                metadata_filter=filter_dict,
                weight=weight,
                tags=(base_name, group_by, str(group_value)),
            )
        )
    return configs


def route_query_to_sources(
    query: str,
    sources: Sequence[SourceRetrieverConfig],
    max_sources: int | None = None,
) -> list[SourceRetrieverConfig]:
    """Heuristically rank and select likely sources for a query."""
    query_terms = _normalize_terms(query)
    if not query_terms:
        selected = list(sources)
    else:
        ranked: list[tuple[tuple[int, float], SourceRetrieverConfig]] = []
        for source in sources:
            source_terms = _source_terms(source)
            overlap = len(query_terms & source_terms)
            score = (overlap, max(source.weight, 0.0))
            ranked.append((score, source))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [source for score, source in ranked if score[0] > 0]
        if not selected:
            selected = [source for _, source in ranked]
        elif max_sources is not None and len(selected) < max_sources:
            seen = {source.name for source in selected}
            for _, source in ranked:
                if source.name in seen:
                    continue
                selected.append(source)
                seen.add(source.name)
                if len(selected) >= max_sources:
                    break

    if max_sources is not None:
        return selected[:max_sources]
    return selected


def plan_query_sources_with_llm(
    query: str,
    sources: Sequence[SourceRetrieverConfig],
    *,
    provider: Provider,
    model: str | None = None,
    api_key: str | None = None,
    max_sources: int | None = None,
) -> QueryRoutingDecision | None:
    """Use an LLM to choose the most relevant sources for a query."""
    if not sources:
        return None

    source_payload = [
        {
            "name": source.name,
            "description": source.description,
            "metadata_filter": source.metadata_filter,
            "tags": list(source.tags),
            "weight": source.weight,
        }
        for source in sources
    ]
    prompt = "\n".join(
        [
            "You are planning retrieval across multiple financial data sources.",
            "Return valid JSON only.",
            "Choose the smallest relevant set of sources needed to answer the question.",
            "If the question explicitly compares sources, select all needed sources.",
            f"Maximum number of sources to return: {max_sources or len(sources)}",
            "",
            f"Question: {query}",
            "",
            "Available Sources:",
            json.dumps(source_payload, indent=2),
            "",
            "JSON schema:",
            json.dumps(
                {
                    "selected_sources": ["source_name"],
                    "reasoning": "short explanation",
                },
                indent=2,
            ),
        ]
    )

    try:
        response = _run_model_request(
            provider=provider,
            model=model,
            api_key=api_key,
            response_format="text",
            prompt=prompt,
            system_prompt="You are a retrieval planning assistant. Output strict JSON only.",
            temperature=0.0,
            max_tokens=400,
            proxy_port=None,
        )
    except Exception:
        return None

    try:
        parsed = _parse_json_object(response.content)
    except ValueError:
        return None

    selected_sources = [
        source_name
        for source_name in parsed.get("selected_sources", [])
        if isinstance(source_name, str)
    ]
    if not selected_sources:
        return None

    if max_sources is not None:
        selected_sources = selected_sources[:max_sources]

    reasoning = str(parsed.get("reasoning", "")).strip()
    return QueryRoutingDecision(
        selected_sources=selected_sources,
        reasoning=reasoning,
        planner_used=True,
    )


def retrieve_multi_source_documents(
    query: str,
    sources: Sequence[SourceRetrieverConfig],
    mode: QueryMode = "ensemble",
    *,
    global_metadata_filter: dict[str, Any] | None = None,
    per_source_metadata_filters: dict[str, dict[str, Any]] | None = None,
    max_sources: int | None = None,
    use_llm_planner: bool = False,
    planner_provider: Provider | None = None,
) -> MultiSourceQueryResult:
    """Retrieve documents across multiple sources using routing, filtering, and ensemble scoring."""
    selected_sources = list(sources)
    routing_decision: QueryRoutingDecision | None = None

    if mode == "routed":
        if use_llm_planner and planner_provider:
            routing_decision = plan_query_sources_with_llm(
                query,
                sources,
                provider=planner_provider,
                max_sources=max_sources,
            )
            if routing_decision is not None:
                selected_sources = _select_sources_by_name(
                    sources,
                    routing_decision.selected_sources,
                    max_sources=max_sources,
                )
        if routing_decision is None:
            selected_sources = route_query_to_sources(query, sources, max_sources=max_sources)
            routing_decision = QueryRoutingDecision(
                selected_sources=[source.name for source in selected_sources],
                reasoning="Keyword overlap fallback.",
                planner_used=False,
            )
    elif max_sources is not None:
        selected_sources = list(sources)[:max_sources]

    source_results: list[SourceRetrievalResult] = []
    for source in selected_sources:
        merged_filter = _merge_filters(
            source.metadata_filter,
            global_metadata_filter,
            (per_source_metadata_filters or {}).get(source.name),
        )
        documents = _invoke_retriever(source.retriever, query)
        if merged_filter:
            documents = filter_documents_by_metadata(documents, merged_filter)
        source_results.append(
            SourceRetrievalResult(
                source_name=source.name,
                documents=documents,
                metadata_filter=merged_filter,
            )
        )

    if mode == "separate":
        combined_documents = [
            document
            for source_result in source_results
            for document in source_result.documents
        ]
    else:
        source_weights = {source.name: source.weight for source in selected_sources}
        combined_documents = _ensemble_documents(source_results, source_weights)

    if routing_decision is None:
        routing_decision = QueryRoutingDecision(
            selected_sources=[source.name for source in selected_sources],
            reasoning="All selected sources used.",
            planner_used=False,
        )

    return MultiSourceQueryResult(
        query=query,
        mode=mode,
        selected_sources=[source.name for source in selected_sources],
        source_results=source_results,
        combined_documents=combined_documents,
        routing_decision=routing_decision,
    )


def build_multi_source_prompt(
    query: str,
    retrieval: MultiSourceQueryResult,
    *,
    separate_by_source: bool | None = None,
) -> str:
    """Build a prompt that synthesizes evidence across routed or ensemble sources."""
    if separate_by_source is None:
        separate_by_source = retrieval.mode != "ensemble"

    lines = [
        "You are an assistant for financial analysis using multiple source-specific contexts.",
        "Compare evidence across sources, note disagreements, and cite the supporting source names inline using [Source: source_name].",
        "If evidence is missing for a source, say so explicitly.",
        "",
        f"Question: {query}",
        "",
        "Retrieved Context:",
    ]

    if retrieval.routing_decision and retrieval.routing_decision.reasoning:
        lines.append(f"Routing rationale: {retrieval.routing_decision.reasoning}")
        lines.append("")

    if separate_by_source:
        for source_result in retrieval.source_results:
            lines.extend(_format_source_section(source_result))
    else:
        for index, document in enumerate(retrieval.combined_documents, start=1):
            lines.extend(_format_document_block(document, index=index))

    lines.extend(
        [
            "",
            "Answer requirements:",
            "- Synthesize across all relevant sources.",
            "- Use inline citations in the form [Source: source_name].",
            "- Highlight conflicts or stale data when sources disagree.",
        ]
    )
    return "\n".join(lines)


def build_source_specific_prompts(
    query: str,
    retrieval: MultiSourceQueryResult,
) -> dict[str, str]:
    """Build one prompt per source so sources can be queried independently."""
    prompts: dict[str, str] = {}
    for source_result in retrieval.source_results:
        prompt_lines = [
            "You are an assistant for financial analysis using a single data source.",
            f"Only use evidence from source: {source_result.source_name}.",
            f"Question: {query}",
            "",
            f"Context from {source_result.source_name}:",
        ]
        prompt_lines.extend(_format_source_section(source_result, include_header=False))
        prompt_lines.extend(
            [
                "",
                "Answer requirements:",
                f"- Use only evidence from {source_result.source_name}.",
                f"- Cite this source inline as [Source: {source_result.source_name}].",
                "- State clearly if this source alone is insufficient.",
            ]
        )
        prompts[source_result.source_name] = "\n".join(prompt_lines)
    return prompts


def collect_source_citations(retrieval: MultiSourceQueryResult) -> list[SourceCitation]:
    """Aggregate retrieved document metadata into source-level citations."""
    citations: list[SourceCitation] = []
    for source_result in retrieval.source_results:
        if not source_result.documents:
            continue

        filenames = sorted({str(doc.metadata.get("filename", "")).strip() for doc in source_result.documents if doc.metadata.get("filename")})
        source_types = sorted({str(doc.metadata.get("source_type", "")).strip() for doc in source_result.documents if doc.metadata.get("source_type")})
        chunk_indices = sorted({int(doc.metadata.get("chunk_index")) for doc in source_result.documents if isinstance(doc.metadata.get("chunk_index"), int)})
        headers: list[str] = []
        for doc in source_result.documents:
            for key in ("Header 1", "Header 2", "Header 3"):
                value = str(doc.metadata.get(key, "")).strip()
                if value and value not in headers:
                    headers.append(value)

        citations.append(
            SourceCitation(
                source_name=source_result.source_name,
                filenames=filenames,
                source_types=source_types,
                chunk_indices=chunk_indices,
                section_headers=headers,
                document_count=len(source_result.documents),
            )
        )
    return citations


def format_source_citations(
    retrieval: MultiSourceQueryResult,
    *,
    response_type: str = "Markdown",
) -> str:
    """Format source-level citations for final answer rendering."""
    citations = collect_source_citations(retrieval)
    if not citations:
        return "No source citations available."

    is_plain_text = response_type == "Plain Text"
    lines: list[str] = ["Sources Consulted:"]
    for citation in citations:
        filename_text = ", ".join(citation.filenames) or "unknown file"
        source_type_text = ", ".join(citation.source_types) or "unknown type"
        chunk_text = ", ".join(str(index) for index in citation.chunk_indices[:6]) or "n/a"
        header_text = ", ".join(citation.section_headers[:3])
        if is_plain_text:
            lines.append(
                f"- {citation.source_name}: files={filename_text}; types={source_type_text}; "
                f"chunks={chunk_text}; documents={citation.document_count}"
            )
            if header_text:
                lines.append(f"  sections={header_text}")
        else:
            line = (
                f"- **{citation.source_name}**: files={filename_text}; types={source_type_text}; "
                f"chunks={chunk_text}; documents={citation.document_count}"
            )
            lines.append(line)
            if header_text:
                lines.append(f"  - sections: {header_text}")
    return "\n".join(lines)


def query_with_multi_source_prompting(
    query: str,
    sources: Sequence[SourceRetrieverConfig],
    *,
    provider: Provider,
    model: str | None = None,
    api_key: str | None = None,
    response_format: str = "text",
    mode: QueryMode = "ensemble",
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    proxy_port: int | None = None,
    global_metadata_filter: dict[str, Any] | None = None,
    per_source_metadata_filters: dict[str, dict[str, Any]] | None = None,
    max_sources: int | None = None,
    use_llm_planner: bool = False,
    planner_provider: Provider | None = None,
    auto_truncate_prompt: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> MultiSourcePromptResult:
    """Retrieve across sources, build prompts, and execute a model request."""
    retrieval = retrieve_multi_source_documents(
        query,
        sources,
        mode=mode,
        global_metadata_filter=global_metadata_filter,
        per_source_metadata_filters=per_source_metadata_filters,
        max_sources=max_sources,
        use_llm_planner=use_llm_planner,
        planner_provider=planner_provider,
    )
    prompt = build_multi_source_prompt(query, retrieval)
    source_prompts = build_source_specific_prompts(query, retrieval)
    prompt = _compress_prompt_for_oversized_requests(
        query,
        retrieval,
        prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt or "You are a helpful financial analysis assistant.",
        temperature=temperature,
        max_tokens=max_tokens,
        proxy_port=proxy_port,
        auto_truncate_prompt=auto_truncate_prompt,
    )
    response = _run_model_request(
        provider=provider,
        model=model,
        api_key=api_key,
        response_format=response_format,
        prompt=prompt,
        system_prompt=system_prompt or "You are a helpful financial analysis assistant.",
        temperature=temperature,
        max_tokens=max_tokens,
        proxy_port=proxy_port,
        auto_truncate_prompt=auto_truncate_prompt,
        tools=tools,
    )
    return MultiSourcePromptResult(
        retrieval=retrieval,
        prompt=prompt,
        source_prompts=source_prompts,
        response=response,
    )


def _compress_prompt_for_oversized_requests(
    query: str,
    retrieval: MultiSourceQueryResult,
    prompt: str,
    *,
    provider: Provider,
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str,
    temperature: float,
    max_tokens: int | None,
    proxy_port: int | None,
    auto_truncate_prompt: bool,
) -> str:
    model_name = model or resolve_model_name(provider)
    safe_budget = get_model_safe_input_budget(model_name)
    if safe_budget is None:
        return prompt

    prompt_tokens = count_message_tokens(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    if prompt_tokens <= safe_budget:
        return prompt

    compressed_summaries = _summarize_retrieval_for_question(
        query,
        retrieval,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        proxy_port=proxy_port,
        auto_truncate_prompt=auto_truncate_prompt,
        safe_budget=safe_budget,
    )
    return _build_compressed_multi_source_prompt(query, retrieval, compressed_summaries)


def _summarize_retrieval_for_question(
    query: str,
    retrieval: MultiSourceQueryResult,
    *,
    provider: Provider,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float,
    max_tokens: int | None,
    proxy_port: int | None,
    auto_truncate_prompt: bool,
    safe_budget: int,
) -> dict[str, list[str]]:
    compressed: dict[str, list[str]] = {}
    for source_result in retrieval.source_results:
        if not source_result.documents:
            compressed[source_result.source_name] = ["No matching documents retrieved."]
            continue

        batch_prompts = _build_summary_batch_prompts(query, source_result, safe_budget)
        source_summaries: list[str] = []
        for batch_prompt in batch_prompts:
            summary_response = _run_model_request(
                provider=provider,
                model=model,
                api_key=api_key,
                response_format="text",
                prompt=batch_prompt,
                system_prompt=OVERSIZED_PROMPT_SUMMARY_SYSTEM_PROMPT,
                temperature=min(temperature, 0.2),
                max_tokens=min(max_tokens, 400) if max_tokens is not None else 400,
                proxy_port=proxy_port,
                auto_truncate_prompt=auto_truncate_prompt,
                tools=None,
            )
            source_summaries.append(summary_response.content.strip() or "Insufficient evidence.")
        compressed[source_result.source_name] = source_summaries or ["Insufficient evidence."]
    return compressed


def _build_summary_batch_prompts(
    query: str,
    source_result: SourceRetrievalResult,
    safe_budget: int,
) -> list[str]:
    base_prompt = "\n".join(
        [
            f"Question: {query}",
            f"Source: {source_result.source_name}",
            "Compress the following retrieved evidence into concise bullet points.",
            "Requirements:",
            "- Keep only evidence relevant to the question.",
            "- Preserve figures, dates, percentages, and directional changes.",
            "- Mention conflicts, caveats, or uncertainty explicitly.",
            "- End with 'Insufficient evidence.' if the batch does not support the question.",
            "",
            "Retrieved Evidence:",
            "",
        ]
    )
    batch_budget = _derive_summary_batch_budget(safe_budget)
    base_tokens = count_message_tokens(
        [
            {"role": "system", "content": OVERSIZED_PROMPT_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": base_prompt},
        ]
    )
    available_block_tokens = max(64, batch_budget - base_tokens)

    prompts: list[str] = []
    current_blocks: list[str] = []
    for index, document in enumerate(source_result.documents, start=1):
        block = "\n".join(_format_document_block(document, index=index)).strip()
        candidate_blocks = current_blocks + [block]
        candidate_prompt = _render_summary_batch_prompt(base_prompt, candidate_blocks)
        candidate_tokens = count_message_tokens(
            [
                {"role": "system", "content": OVERSIZED_PROMPT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": candidate_prompt},
            ]
        )
        if current_blocks and candidate_tokens > batch_budget:
            prompts.append(_render_summary_batch_prompt(base_prompt, current_blocks))
            current_blocks = []

        trimmed_block = trim_text_to_tokens(block, available_block_tokens)
        if trimmed_block:
            current_blocks.append(trimmed_block)

    if current_blocks:
        prompts.append(_render_summary_batch_prompt(base_prompt, current_blocks))

    return prompts or [base_prompt + "No retrieved evidence."]


def _derive_summary_batch_budget(safe_budget: int) -> int:
    return max(128, min(safe_budget - 64, int(safe_budget * 0.65)))


def _render_summary_batch_prompt(base_prompt: str, blocks: Sequence[str]) -> str:
    return base_prompt + "\n\n".join(blocks)


def _build_compressed_multi_source_prompt(
    query: str,
    retrieval: MultiSourceQueryResult,
    compressed_summaries: dict[str, list[str]],
) -> str:
    lines = [
        "You are an assistant for financial analysis using compressed multi-source evidence.",
        "Compare evidence across sources, note disagreements, and cite the supporting source names inline using [Source: source_name].",
        "If evidence is missing for a source, say so explicitly.",
        "",
        f"Question: {query}",
        "",
        "Compressed Source Summaries:",
    ]

    if retrieval.routing_decision and retrieval.routing_decision.reasoning:
        lines.append(f"Routing rationale: {retrieval.routing_decision.reasoning}")
        lines.append("")

    for source_result in retrieval.source_results:
        lines.append(f"### Source: {source_result.source_name}")
        source_summaries = compressed_summaries.get(source_result.source_name, [])
        for batch_index, summary in enumerate(source_summaries, start=1):
            prefix = f"Batch {batch_index}: " if len(source_summaries) > 1 else ""
            lines.append(f"- {prefix}{summary}")
        if not source_summaries:
            lines.append("- No summary available.")
        lines.append("")

    lines.extend(
        [
            "Answer requirements:",
            "- Synthesize across all relevant sources.",
            "- Use inline citations in the form [Source: source_name].",
            "- Highlight conflicts or stale data when sources disagree.",
        ]
    )
    return "\n".join(lines)


def _invoke_retriever(retriever: RetrieverLike | Any, query: str) -> list[Document]:
    if hasattr(retriever, "invoke"):
        documents = retriever.invoke(query)
    elif hasattr(retriever, "get_relevant_documents"):
        documents = retriever.get_relevant_documents(query)
    else:
        raise TypeError("Retriever must expose invoke() or get_relevant_documents().")
    return list(documents)


def _run_model_request(
    *,
    provider: Provider,
    response_format: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int | None,
    proxy_port: int | None,
    model: str | None = None,
    api_key: str | None = None,
    auto_truncate_prompt: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> ModelResponse:
    from .request import ModelRequest, RequestPayload

    return ModelRequest(provider=provider, format=response_format, model=model, api_key=api_key).request(
        RequestPayload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            proxy_port=proxy_port,
            auto_truncate_prompt=auto_truncate_prompt,
            tools=tools,
        )
    )


def _merge_filters(*filters: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metadata_filter in filters:
        if metadata_filter:
            merged.update(metadata_filter)
    return merged


def _ensemble_documents(
    source_results: Sequence[SourceRetrievalResult],
    source_weights: dict[str, float],
) -> list[Document]:
    scores: dict[tuple[str, str], float] = defaultdict(float)
    doc_lookup: dict[tuple[str, str], Document] = {}

    for source_result in source_results:
        weight = max(source_weights.get(source_result.source_name, 1.0), 0.0)
        for rank, document in enumerate(source_result.documents):
            identity = _document_identity(document)
            scores[identity] += weight / (rank + 1)
            doc_lookup[identity] = document

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_lookup[identity] for identity, _ in ranked]


def _document_identity(document: Document) -> tuple[str, str]:
    source = str(document.metadata.get("source", ""))
    chunk = str(document.metadata.get("chunk_index", ""))
    filename = str(document.metadata.get("filename", ""))
    return (f"{source}:{filename}:{chunk}", document.page_content)


def _discover_metadata_groups(vector_store: Any, metadata_key: str) -> list[str]:
    docstore = getattr(vector_store, "docstore", None)
    stored_docs = getattr(docstore, "_dict", {}) if docstore is not None else {}
    values = {
        str(document.metadata.get(metadata_key)).strip()
        for document in stored_docs.values()
        if isinstance(document, Document) and document.metadata.get(metadata_key)
    }
    values.discard("")
    return sorted(values)


def _select_sources_by_name(
    sources: Sequence[SourceRetrieverConfig],
    selected_names: Sequence[str],
    *,
    max_sources: int | None = None,
) -> list[SourceRetrieverConfig]:
    selected: list[SourceRetrieverConfig] = []
    selected_set = set(selected_names)
    for source in sources:
        if source.name in selected_set:
            selected.append(source)
    if max_sources is not None and len(selected) < max_sources:
        seen = {source.name for source in selected}
        for source in sources:
            if source.name in seen:
                continue
            selected.append(source)
            seen.add(source.name)
            if len(selected) >= max_sources:
                break
    return selected[:max_sources] if max_sources is not None else selected


def _source_terms(source: SourceRetrieverConfig) -> set[str]:
    terms: set[str] = set()
    terms |= _normalize_terms(source.name)
    terms |= _normalize_terms(source.description)
    for tag in source.tags:
        terms |= _normalize_terms(tag)
    for key, value in source.metadata_filter.items():
        terms |= _normalize_terms(key)
        terms |= _normalize_terms(str(value))
    return terms


def _normalize_terms(text: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    return {term for term in normalized.split() if len(term) >= 2}


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(line for line in lines if not line.strip().startswith("```"))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(stripped[start : end + 1])


def _format_source_section(
    source_result: SourceRetrievalResult,
    *,
    include_header: bool = True,
) -> list[str]:
    lines: list[str] = []
    if include_header:
        lines.append(f"### Source: {source_result.source_name}")
        if source_result.metadata_filter:
            lines.append(f"Metadata Filter: {source_result.metadata_filter}")
    if not source_result.documents:
        lines.append("- No matching documents retrieved.")
        lines.append("")
        return lines
    for index, document in enumerate(source_result.documents, start=1):
        lines.extend(_format_document_block(document, index=index))
    lines.append("")
    return lines


def _format_document_block(document: Document, *, index: int) -> list[str]:
    metadata = document.metadata or {}
    metadata_summary = ", ".join(
        f"{key}={value}"
        for key, value in metadata.items()
        if key in {"source", "source_type", "filename", "chunk_index", "Header 1", "Header 2", "Header 3"}
    )
    lines = [f"[{index}] {metadata_summary}".rstrip()]
    lines.append(document.page_content)
    lines.append("")
    return lines
