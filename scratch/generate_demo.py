import logging

from docsrag.llm import OllamaGenerator

logging.basicConfig(level=logging.INFO)
gen = OllamaGenerator()
r = gen.generate('Explain INT8 quantization in two sentences.', system='You are a concise ML engineer.')
print(r.text)
print(r)