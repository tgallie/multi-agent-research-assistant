"""Web search providers with normalized, citation-ready output."""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, HttpUrl

from agent.schemas import Source, ToolResult, ToolStatus
from agent.tools.base import sanitize_text


class WebSearchArgs(BaseModel):
    """Validated web-search arguments."""

    query: str = Field(min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=8)


class SearchRecord(BaseModel):
    """Provider-independent web result record."""

    title: str
    url: HttpUrl
    snippet: str


class WebSearchTool:
    """Search Tavily or SerpAPI and normalize result provenance."""

    name = "web_search"
    args_schema = WebSearchArgs

    def __init__(
        self,
        *,
        provider: Literal["tavily", "serpapi"],
        api_key: str,
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure one explicit provider; credentials are never serialized."""

        if not api_key.strip():
            raise ValueError("web search API key is required")
        self.provider = provider
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def invoke(self, arguments: BaseModel) -> ToolResult:
        """Run a search and return only validated records."""

        args = WebSearchArgs.model_validate(arguments)
        started = time.perf_counter()
        response = self._request(args)
        response.raise_for_status()
        records = self._parse(response.json(), args.max_results)
        sources = [
            Source(
                id=f"src_{sha256(str(record.url).encode()).hexdigest()[:10]}",
                title=sanitize_text(record.title, limit=300),
                url=record.url,
                snippet=sanitize_text(record.snippet, limit=4_000),
            )
            for record in records
            if record.snippet.strip()
        ]
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            content="\n".join(f"[{item.id}] {item.snippet}" for item in sources),
            sources=sources,
            latency_ms=int((time.perf_counter() - started) * 1_000),
        )

    def _request(self, args: WebSearchArgs) -> httpx.Response:
        if self.provider == "tavily":
            return self._client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._api_key,
                    "query": args.query,
                    "max_results": args.max_results,
                    "search_depth": "advanced",
                },
            )
        return self._client.get(
            "https://serpapi.com/search.json",
            params={"api_key": self._api_key, "q": args.query, "num": args.max_results},
        )

    def _parse(self, payload: Any, limit: int) -> list[SearchRecord]:
        if not isinstance(payload, dict):
            raise ValueError("search provider returned a non-object response")
        raw_results = payload.get("results" if self.provider == "tavily" else "organic_results")
        if not isinstance(raw_results, list):
            raise ValueError("search provider response omitted result list")
        records: list[SearchRecord] = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            try:
                records.append(
                    SearchRecord(
                        title=item.get("title", "Untitled source"),
                        url=item.get("url") or item.get("link"),
                        snippet=item.get("content") or item.get("snippet") or "",
                    )
                )
            except (ValueError, TypeError):
                continue
        return records
