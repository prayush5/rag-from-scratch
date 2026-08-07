import re

text = "Hello world. This is Ram. How are you?"
sentences = re.split(r'(?<=[.!?])\s+', text)

current_chunk = ""
chunks = []

for sentence in sentences:
    candidate = current_chunk + " " + sentence

    if len(candidate) <= 30:
        current_chunk = candidate
    
    else:
        chunks.append(current_chunk)
        current_chunk = sentence

if current_chunk:
    chunks.append(current_chunk)

print(chunks)