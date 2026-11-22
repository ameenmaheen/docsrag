import os
import time

import ollama
from dotenv import load_dotenv

load_dotenv()

t0 = time.perf_counter()
resp = ollama.chat(
    model=os.environ['LLM_MODEL'],
    messages=[
        {'role': 'system', 'content': 'You are a terse Android performance expert.'},
        {'role': 'user', 'content': 'In two sentences, what is quantization in ML?'},
    ],
    options={'num_predict': 200, 'temperature': 0},
)
print(resp)                          # full object
print(resp.message.content)          # just the text
print(resp.prompt_eval_count, resp.eval_count, resp.done_reason)
print('latency:', round(time.perf_counter() - t0, 2), 's')