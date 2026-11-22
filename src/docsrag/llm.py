from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
import ollama
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    load_s: float
    tokens_per_s: float
    stop_reason: str | None


class Generator(Protocol):
    def generate(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> GenerationResult: ...


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        max_tokens: int = 15,
    ) -> None:
        self.model = model or os.getenv('LLM_MODEL', 'llama3.2:3b')
        url = url or os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self._client = ollama.Client(host=url, timeout=timeout_s)
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_tokens = max_tokens

    def generate(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> GenerationResult:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        resp, latency_s = self._chat_with_retry(messages, temperature)
        result = self._to_result(resp, latency_s)
        log.info(
            'model=%s in=%d out=%d latency=%.2fs tok/s=%.1f',
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.latency_s,
            result.tokens_per_s,
        )
        return result

    def _chat_with_retry(self, messages: list[dict], temperature: float):
        options = {'temperature': temperature, 'num_predict': self.max_tokens}
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                resp = self._client.chat(
                    model=self.model, messages=messages, options=options
                )
                return resp, time.perf_counter() - start
            except ollama.ResponseError as e:
                # 4xx means our request is wrong (e.g. model not found): retrying won't help
                if e.status_code < 500 or attempt == self.max_retries:
                    raise
            except (ConnectionError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
            wait = self.backoff_s * 2 ** (attempt - 1)
            log.warning('attempt %d failed, retrying in %.1fs', attempt, wait)
            time.sleep(wait)
        raise RuntimeError('unreachable')

    @staticmethod
    def _to_result(resp, latency_s: float) -> GenerationResult:
        completion = resp.eval_count or 0
        eval_s = (resp.eval_duration or 0) / 1e9
        return GenerationResult(
            text=resp.message.content or '',
            model=resp.model,
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=completion,
            latency_s=latency_s,
            load_s=(resp.load_duration or 0) / 1e9,
            tokens_per_s=completion / eval_s if eval_s > 0 else 0.0,
            stop_reason=resp.done_reason,
        )