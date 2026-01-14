"""Unit tests for the teacher client (cache, cost meter, retries).

We don't hit the OpenAI API here; the network calls are stubbed out via
monkey-patching the ``AsyncOpenAI`` instance on the client.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import RateLimitError

from distill_sql.config import TeacherConfig
from distill_sql.teacher.client import (
    BudgetExceededError,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CostMeter,
    DiskCache,
    StubTeacherClient,
    TeacherClient,
    build_teacher_messages,
    request_dict,
)


def _req(model: str = "gpt-4o-mini", text: str = "hi", idx: int = 0) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=(ChatMessage("user", text),),
        temperature=0.3,
        max_tokens=128,
        sample_index=idx,
    )


def test_cache_key_stable_under_field_order() -> None:
    a = _req(text="hello")
    b = _req(text="hello")
    assert a.cache_key() == b.cache_key()


def test_cache_key_differs_by_sample_index() -> None:
    assert _req(idx=0).cache_key() != _req(idx=1).cache_key()


def test_cache_key_differs_by_model() -> None:
    assert _req(model="gpt-4o").cache_key() != _req(model="gpt-4o-mini").cache_key()


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = "deadbeef" * 4
    assert cache.get(key) is None
    rsp = CompletionResponse(
        text="answer",
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
        model="gpt-4o-mini",
        cached=False,
    )
    cache.put(key, rsp)
    out = cache.get(key)
    assert out is not None
    assert out.text == "answer"
    assert out.cached is True


def test_disk_cache_uses_two_level_fanout(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = "abcdef" * 8
    cache.put(key, CompletionResponse("x", 1, 1, "stop", "m", False))
    assert (tmp_path / "ab" / f"{key}.json").exists()


def test_cost_meter_persists_and_reloads(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path)
    a = CostMeter.load(cfg)
    a.record(1_000_000, 1_000_000)  # 0.15 + 0.60 = 0.75 USD
    assert a.spend_usd == pytest.approx(0.75)
    b = CostMeter.load(cfg)
    assert b.spend_usd == pytest.approx(0.75)


def test_cost_meter_would_exceed(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path, max_spend_usd=0.5)
    m = CostMeter.load(cfg)
    # 1M input + 1M output = $0.75 -> over the $0.50 cap.
    assert m.would_exceed(prompt_tokens=1_000_000, max_completion=1_000_000)
    # Tiny call stays under the cap.
    assert not m.would_exceed(prompt_tokens=100, max_completion=100)


def test_cost_meter_remaining(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path, max_spend_usd=10.0)
    m = CostMeter.load(cfg)
    m.record(1_000_000, 1_000_000)
    assert m.remaining() == pytest.approx(9.25)


@pytest.mark.asyncio
async def test_client_returns_cached_when_present(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path / "cache")
    client = TeacherClient(cfg, api_key="fake")
    cached = CompletionResponse(
        text="from cache",
        prompt_tokens=1,
        completion_tokens=1,
        finish_reason="stop",
        model="gpt-4o-mini",
        cached=False,
    )
    req = _req()
    client.cache.put(req.cache_key(), cached)
    out = await client.complete(req)
    assert out.text == "from cache"
    assert out.cached is True


@pytest.mark.asyncio
async def test_client_records_spend(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path / "cache")
    client = TeacherClient(cfg, api_key="fake")

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="SELECT 1"), finish_reason="stop")]
    fake_resp.usage = MagicMock(prompt_tokens=20, completion_tokens=10)

    client._client.chat.completions.create = AsyncMock(return_value=fake_resp)  # type: ignore[method-assign]
    out = await client.complete(_req())
    assert out.text == "SELECT 1"
    assert out.cached is False
    assert client.cost.spend_usd > 0


@pytest.mark.asyncio
async def test_client_raises_on_budget_exceeded(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path / "cache", max_spend_usd=0.0)
    client = TeacherClient(cfg, api_key="fake")
    with pytest.raises(BudgetExceededError):
        await client.complete(_req(text="x" * 5000))


@pytest.mark.asyncio
async def test_client_retries_on_rate_limit(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path / "cache", max_retries=2)
    client = TeacherClient(cfg, api_key="fake")

    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="OK"), finish_reason="stop")]
    ok.usage = MagicMock(prompt_tokens=2, completion_tokens=2)

    err = RateLimitError(
        "rate limited",
        response=MagicMock(status_code=429),
        body=None,
    )

    create = AsyncMock(side_effect=[err, ok])
    client._client.chat.completions.create = create  # type: ignore[method-assign]
    out = await client.complete(_req())
    assert out.text == "OK"
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_stub_client_complete_uses_canned_response() -> None:
    req = _req()
    stub = StubTeacherClient({req.cache_key(): "stubbed"})
    out = await stub.complete(req)
    assert out.text == "stubbed"
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_stub_client_default_fallback() -> None:
    stub = StubTeacherClient({"default": "FALLBACK"})
    out = await stub.complete(_req(text="anything"))
    assert out.text == "FALLBACK"


@pytest.mark.asyncio
async def test_client_gather_dispatches_in_parallel(tmp_path: Path) -> None:
    cfg = TeacherConfig(cache_dir=tmp_path / "cache", concurrency=2)
    client = TeacherClient(cfg, api_key="fake")

    call_times: list[float] = []

    async def fake_create(**_kwargs: object) -> object:
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.05)
        rsp = MagicMock()
        rsp.choices = [MagicMock(message=MagicMock(content="x"), finish_reason="stop")]
        rsp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        return rsp

    client._client.chat.completions.create = AsyncMock(side_effect=fake_create)  # type: ignore[method-assign]
    reqs = [_req(idx=i) for i in range(4)]
    await client.gather(reqs)
    # 4 calls × 0.05s, concurrency 2 -> wall-clock ~0.10s, not 0.20s.
    span = max(call_times) - min(call_times)
    assert span < 0.15


def test_build_teacher_messages_yields_system_then_user() -> None:
    msgs = build_teacher_messages("sys", "usr")
    assert msgs[0].role == "system" and msgs[0].content == "sys"
    assert msgs[1].role == "user" and msgs[1].content == "usr"


def test_request_dict_has_cache_key() -> None:
    d = request_dict(_req())
    assert "cache_key" in d
    assert d["model"] == "gpt-4o-mini"
