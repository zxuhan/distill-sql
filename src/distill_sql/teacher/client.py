"""Async OpenAI client wrapper with disk caching, retries, and a cost meter.

The pipeline calls into ``TeacherClient.complete()`` for every (question, mode,
sample_index) tuple. Responses are keyed by a SHA-256 hash of the request and
written to disk so re-runs are free, and a structured cost log is written
alongside so we can audit cumulative spend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ..config import TeacherConfig

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    """Minimal subset of OpenAI's chat message: role + content."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class CompletionRequest:
    """Inputs to the teacher; the cache key is derived from this."""

    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float
    max_tokens: int
    sample_index: int = 0
    """Used to vary cache keys across self-consistency samples."""

    def cache_key(self) -> str:
        payload = {
            "model": self.model,
            "messages": [m.as_dict() for m in self.messages],
            "temperature": round(self.temperature, 4),
            "max_tokens": self.max_tokens,
            "sample_index": self.sample_index,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CompletionResponse:
    """What we save and downstream code reads."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    model: str
    cached: bool


# ---------------------------------------------------------------------------
# Cost meter
# ---------------------------------------------------------------------------


@dataclass
class CostMeter:
    """Tracks cumulative spend with a hard cap.

    Persists to a JSON sidecar so spending across multiple runs accumulates;
    the cap is enforced *before* a request is dispatched.
    """

    cap_usd: float
    price_input_per_m: float
    price_output_per_m: float
    sidecar_path: Path
    spend_usd: float = 0.0

    @classmethod
    def load(cls, cfg: TeacherConfig) -> CostMeter:
        path = cfg.cache_dir / "cost.json"
        meter = cls(
            cap_usd=cfg.max_spend_usd,
            price_input_per_m=cfg.price_input_per_m,
            price_output_per_m=cfg.price_output_per_m,
            sidecar_path=path,
        )
        if path.exists():
            data = json.loads(path.read_text())
            meter.spend_usd = float(data.get("spend_usd", 0.0))
        return meter

    def estimate_call(self, prompt_tokens: int, max_completion: int) -> float:
        """Upper-bound estimate of a call's cost (assumes max-out completion)."""
        return (
            prompt_tokens * self.price_input_per_m / 1_000_000
            + max_completion * self.price_output_per_m / 1_000_000
        )

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost = (
            prompt_tokens * self.price_input_per_m / 1_000_000
            + completion_tokens * self.price_output_per_m / 1_000_000
        )
        self.spend_usd += cost
        self.persist()

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spend_usd)

    def would_exceed(self, prompt_tokens: int, max_completion: int) -> bool:
        return self.spend_usd + self.estimate_call(prompt_tokens, max_completion) > self.cap_usd

    def persist(self) -> None:
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_path.write_text(
            json.dumps(
                {"spend_usd": round(self.spend_usd, 6), "cap_usd": self.cap_usd},
                indent=2,
            ),
        )


class BudgetExceededError(RuntimeError):
    """Raised when a call would push cumulative spend past the cap."""


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class DiskCache:
    """Tiny content-addressed cache. Each request maps to one JSON file."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Two-level fanout to keep any one directory small.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CompletionResponse | None:
        p = self._path(key)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return CompletionResponse(
            text=data["text"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            finish_reason=data.get("finish_reason"),
            model=data["model"],
            cached=True,
        )

    def put(self, key: str, response: CompletionResponse) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "text": response.text,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "finish_reason": response.finish_reason,
                    "model": response.model,
                    "saved_at": time.time(),
                },
            ),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


class TeacherClient:
    """High-level wrapper. Holds the AsyncOpenAI, the cache, and the cost meter."""

    def __init__(
        self,
        cfg: TeacherConfig,
        api_key: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.cache = DiskCache(cfg.cache_dir)
        self.cost = CostMeter.load(cfg)
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=cfg.request_timeout_s,
            max_retries=0,  # we wrap with tenacity ourselves
        )
        self._sem = asyncio.Semaphore(cfg.concurrency)

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Return a completion, hitting the cache where possible."""
        cached = self.cache.get(req.cache_key())
        if cached is not None:
            return cached

        # Estimate: cost meter uses prompt-token approximation (4 chars/token).
        approx_prompt_tokens = sum(len(m.content) for m in req.messages) // 4
        if self.cost.would_exceed(approx_prompt_tokens, req.max_tokens):
            raise BudgetExceededError(
                f"refusing call: estimated cumulative spend "
                f"${self.cost.spend_usd:.4f} + "
                f"${self.cost.estimate_call(approx_prompt_tokens, req.max_tokens):.4f} "
                f"exceeds cap ${self.cost.cap_usd:.2f}",
            )

        async with self._sem:
            text, prompt_tokens, completion_tokens, finish_reason = await self._call(req)

        self.cost.record(prompt_tokens, completion_tokens)
        response = CompletionResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            model=req.model,
            cached=False,
        )
        self.cache.put(req.cache_key(), response)
        return response

    async def _call(
        self,
        req: CompletionRequest,
    ) -> tuple[str, int, int, str | None]:
        """Single-request path with tenacity retries on transient OpenAI errors."""
        retry = AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_random_exponential(min=1, max=30),
            retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
            reraise=True,
        )
        async for attempt in retry:
            with attempt:
                resp = await self._client.chat.completions.create(
                    model=req.model,
                    messages=[m.as_dict() for m in req.messages],  # type: ignore[arg-type]
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    n=1,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                usage = resp.usage
                pt = int(usage.prompt_tokens) if usage else 0
                ct = int(usage.completion_tokens) if usage else 0
                return content, pt, ct, choice.finish_reason
        raise RuntimeError("retry loop exited without value")  # pragma: no cover

    async def gather(
        self,
        requests: Iterable[CompletionRequest],
    ) -> list[CompletionResponse]:
        """Run a batch of requests concurrently, respecting the semaphore."""
        return list(await asyncio.gather(*(self.complete(r) for r in requests)))

    def remaining_budget(self) -> float:
        return self.cost.remaining()


# ---------------------------------------------------------------------------
# Stub client for testing
# ---------------------------------------------------------------------------


class StubTeacherClient:
    """In-memory stub used in tests. Returns canned responses by request hash.

    Mirrors the shape of TeacherClient.complete() so tests can swap it in.
    """

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls = 0

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        text = self._responses.get(req.cache_key(), self._responses.get("default", ""))
        return CompletionResponse(
            text=text,
            prompt_tokens=10,
            completion_tokens=20,
            finish_reason="stop",
            model=req.model,
            cached=False,
        )

    async def gather(
        self,
        requests: Iterable[CompletionRequest],
    ) -> list[CompletionResponse]:
        return [await self.complete(r) for r in requests]

    def remaining_budget(self) -> float:
        return 0.0


def build_teacher_messages(
    system_prompt: str,
    user_prompt: str,
) -> Sequence[ChatMessage]:
    """Helper used by the trace pipeline to build the OpenAI message list."""
    return (
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    )


def request_dict(req: CompletionRequest) -> dict[str, Any]:
    """For logging / debugging only. Not used in the main code path."""
    return {
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "sample_index": req.sample_index,
        "n_messages": len(req.messages),
        "cache_key": req.cache_key(),
    }
