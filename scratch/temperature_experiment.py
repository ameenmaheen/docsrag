from docsrag.llm import OllamaGenerator
from docsrag.metrics import summarize

PROMPT = 'Explain in two sentences why INT8 quantization speeds up inference on mobile CPUs.'
RUNS = 10

gen = OllamaGenerator(max_tokens=120)
gen.generate('warm up', max_tokens=5)  # load the model once so load time does not pollute the numbers

settings = [
    ('temperature 0', {'temperature': 0.0}),
    ('temperature 1.0', {'temperature': 1.0}),
    ('temperature 1.0, seed 42', {'temperature': 1.0, 'seed': 42}),
]

for label, kwargs in settings:
    results = [gen.generate(PROMPT, **kwargs) for _ in range(RUNS)]
    stats = summarize(results)
    print(f'== {label}')
    print(f'   distinct outputs : {stats.distinct_outputs}/{stats.n}')
    print(f'   output tokens    : mean {stats.mean_tokens:.1f}, stdev {stats.stdev_tokens:.1f}')
    print(f'   latency          : mean {stats.mean_latency_s:.2f}s, stdev {stats.stdev_latency_s:.2f}s')
    print(f'   speed            : {stats.mean_tokens_per_s:.1f} tok/s')
    print('   first output     :', ' '.join(results[0].text.split())[:150])