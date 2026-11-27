from types import SimpleNamespace

import httpx
import pytest

from docsrag import llm
from docsrag.llm import OllamaGenerator


def fake_resp():
    return SimpleNamespace(
        message=SimpleNamespace(content='hello'),
        model='fake-model',
        prompt_eval_count=5,
        eval_count=10,
        eval_duration=1_000_000_000,
        load_duration=0,
        done_reason='stop',
    )


class FlakyClient:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0
        self.last_kwargs = None

    def chat(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self.failures:
            raise httpx.ReadTimeout('slow')
        return fake_resp()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, 'sleep', lambda s: None)


def test_retries_then_succeeds():
    gen = OllamaGenerator(model='x', max_retries=3)
    gen._client = FlakyClient(failures=2)
    result = gen.generate('hi')
    assert result.text == 'hello'
    assert gen._client.calls == 3


def test_gives_up_after_max_retries():
    gen = OllamaGenerator(model='x', max_retries=2)
    gen._client = FlakyClient(failures=5)
    with pytest.raises(httpx.ReadTimeout):
        gen.generate('hi')
    assert gen._client.calls == 2


def test_system_prompt_is_sent_first():
    gen = OllamaGenerator(model='x')
    gen._client = FlakyClient(failures=0)
    gen.generate('question', system='be brief')
    msgs = gen._client.last_kwargs['messages']
    assert msgs[0] == {'role': 'system', 'content': 'be brief'}
    assert msgs[1]['role'] == 'user'


def test_tokens_per_second():
    gen = OllamaGenerator(model='x')
    gen._client = FlakyClient(failures=0)
    result = gen.generate('hi')
    assert result.tokens_per_s == pytest.approx(10.0)
    assert result.completion_tokens == 10   

def test_options_include_seed_and_stop_when_given():
    gen = OllamaGenerator(model='x')
    gen._client = FlakyClient(failures=0)
    gen.generate('hi', temperature=0.7, seed=42, stop=['END'], max_tokens=33)
    options = gen._client.last_kwargs['options']
    assert options == {'temperature': 0.7, 'num_predict': 33, 'seed': 42, 'stop': ['END']}


def test_options_omit_seed_and_stop_by_default():
    gen = OllamaGenerator(model='x', max_tokens=100)
    gen._client = FlakyClient(failures=0)
    gen.generate('hi')
    assert gen._client.last_kwargs['options'] == {'temperature': 0.0, 'num_predict': 100}


def test_model_comes_from_env(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'env-model')
    assert OllamaGenerator().model == 'env-model'


def test_model_falls_back_to_default(monkeypatch):
    monkeypatch.delenv('LLM_MODEL', raising=False)
    assert OllamaGenerator().model == 'llama3.2:3b'