import time

import httpx

body = {
    'model':'llama3.2:3b',
    'stream':False,
    'messages':[{'role':'user','content':'In two sentences, what is quantization in ML?'}],
    'options':{'num_predict':200,'temperature':0},
}

t0 =time.perf_counter()
r = httpx.post('http://localhost:11434/api/chat',json=body,timeout=120)
print('status',r.status_code,'latency:',round(time.perf_counter()- t0,2),'s')
print('out >>>>',r.json())