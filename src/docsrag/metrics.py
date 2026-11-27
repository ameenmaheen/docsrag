from __future__ import annotations

import statistics
from dataclasses import dataclass

from docsrag.llm import GenerationResult


@dataclass(frozen=True)
class PriceTable:
    input_per_mtok : float = 0.0
    output_per_mtok : float = 0.0


def estimate_cost(result: GenerationResult, prices: PriceTable) -> float:
    return (
        result.prompt_tokens * prices.input_per_mtok 
        + result.completion_tokens * prices.output_per_mtok
    ) / 1_000_000

@dataclass(frozen=True)
class RunStats:
    n : int
    distinct_outputs:int
    mean_tokens : float
    stdev_tokens : float
    mean_latency_s : float
    stdev_latency_s : float
    mean_tokens_per_s : float
    total_cost : float

    @property
    def distinct_ration(self) -> float:
        return self.distinct_outputs / self.n

def summarize(
    results: list[GenerationResult], prices: PriceTable | None = None
) -> RunStats:
    if not results:
        raise ValueError('need at least one result')
    prices = prices or PriceTable()
    tokens = [r.completion_tokens for r in results]
    latencies = [r.latency_s for r in results]
    speeds = [r.tokens_per_s for r in results]
    return RunStats(
        n=len(results),
        distinct_outputs=len({r.text.strip() for r in results}),
        mean_tokens=statistics.fmean(tokens),
        stdev_tokens=statistics.pstdev(tokens),
        mean_latency_s=statistics.fmean(latencies),
        stdev_latency_s=statistics.pstdev(latencies),
        mean_tokens_per_s=statistics.fmean(speeds),
        total_cost=sum(estimate_cost(r, prices) for r in results),
    )