"""
Maestro is a drop-in OpenAI endpoint — point the official SDK at it and route.

  pip install openai
  npx maestro serve                 # in another terminal (mock provider, no keys)
  python examples/openai-sdk.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

resp = client.chat.completions.create(
    model="maestro-auto",  # also: "maestro-fugu", or a real model id for passthrough
    messages=[{"role": "user", "content": "Write a Python function to reverse a linked list."}],
)

print(resp.choices[0].message.content)

# The transparency block is attached to the raw response.
maestro = resp.model_extra.get("maestro") if hasattr(resp, "model_extra") else None
if maestro:
    print("\n--- maestro ---")
    print("route:  ", [t["model"] + " -> " + str(t.get("verdict")) for t in maestro["route"]])
    print("cost:   ", maestro["cost_usd"], "USD")
    print("savings:", maestro["savings_pct"], "%")
