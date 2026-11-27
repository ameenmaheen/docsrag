import pytest

from docsrag.llm import GenerationResult
from docsrag.metrics import PriceTable, estimate_cost, summarize


def make_result(text='a', out=10, inp=5, latency=1.0, tps=10.0):
    return GenerationResult(
        text=text,
        model='m',
        prompt_tokens=inp,
        completion_tokens=out,
        latency_s=latency,
        load_s=0.0,
        tokens_per_s=tps,
        stop_reason='stop',
    )


def test_estimate_cost_is_zero_for_local_prices():
    assert estimate_cost(make_result(), PriceTable()) == 0.0


def test_estimate_cost_with_prices():
    prices = PriceTable(input_per_mtok=1.0, output_per_mtok=2.0)
    result = make_result(inp=1_000_000, out=500_000)
    # 1.0 for the input tokens + 1.0 for the output tokens
    assert estimate_cost(result, prices) == pytest.approx(2.0)


@pytest.mark.parametrize(
    'texts, expected',
    [
        (['a', 'a', 'a'], 1),
        (['a', 'b', 'c'], 3),
        (['a', 'b', 'a'], 2),
        (['a', 'a ', 'a'], 1),
    ],
)
def test_distinct_outputs(texts, expected):
    stats = summarize([make_result(t) for t in texts])
    assert stats.distinct_outputs == expected


def test_mean_and_stdev_of_tokens():
    stats = summarize([make_result(out=10), make_result(out=20)])
    assert stats.mean_tokens == pytest.approx(15.0)
    assert stats.stdev_tokens == pytest.approx(5.0)


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])