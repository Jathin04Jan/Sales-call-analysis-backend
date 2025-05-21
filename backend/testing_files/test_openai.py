#!/usr/bin/env python3
from dotenv import load_dotenv
import os
from openai import OpenAI

# 1) Load your .env
load_dotenv()

# 2) Grab the key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")
print("KEY LOADED:", os.getenv("OPENAI_API_KEY", None))

# 3) Instantiate the new client
client = OpenAI(api_key=api_key)

# 4) Send a simple chat message
response = client.chat.completions.create(
    model="gpt-3.5-turbo",            # or “o4-mini” if your account supports it
    messages=[{"role": "user", "content": "Hello!"}]
)

# 5) Print out the assistant’s reply
print("LLM replied:", response.choices[0].message.content.strip())