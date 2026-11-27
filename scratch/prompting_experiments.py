from docsrag.llm import OllamaGenerator

gen = OllamaGenerator(max_tokens=150)

# A. Same question, different system prompts
QUESTION = 'Why does my Android app jank when I run a TFLite model on the main thread?'
SYSTEMS = {
    'no system prompt': None,
    'terse': 'Answer in at most two sentences. No preamble.',
    'format': 'Answer as exactly three bullet points, each under 12 words.',
    'persona': 'You are a senior Android performance engineer. Be direct and practical.',
}
for name, system in SYSTEMS.items():
    r = gen.generate(QUESTION, system=system)
    print(f'--- {name}: {r.completion_tokens} tokens, {r.latency_s:.1f}s, stop={r.stop_reason}')
    print(r.text)

# B. Few-shot classification
FEW_SHOT = '''Classify each note as perf, memory, power or other. Reply with the label only.

Note: Frame time spikes when the delegate initializes.
Label: perf

Note: Bitmap cache grows until the app is killed.
Label: memory

Note: Wake locks keep the CPU awake overnight.
Label: power

Note: {note}
Label:'''
NOTES = [
    'Inference takes 300 ms on the CPU path.',
    'The model file doubles APK size.',
    'The radio stays on while we poll the server.',
    'The settings screen has a typo.',
]
print('=== few-shot')
for note in NOTES:
    r = gen.generate(FEW_SHOT.format(note=note), max_tokens=5)
    print(f'{r.text.strip():8} <- {note}')

# C. Stop sequences and max tokens
print('=== stop and length')
r = gen.generate('List five Android ML frameworks, one per line, numbered.', stop=['4.'])
print(r.text, '| stop_reason =', r.stop_reason)
r = gen.generate('Explain quantization in detail.', max_tokens=15)
print(r.text, '| stop_reason =', r.stop_reason)