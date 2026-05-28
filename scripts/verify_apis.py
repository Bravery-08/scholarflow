# scripts/verify_apis.py
from core.config import settings
from openai import OpenAI
from groq import Groq
import google.generativeai as genai

print("Testing OpenRouter...")
or_client = OpenAI(api_key=settings.openrouter_api_key,
                   base_url="https://openrouter.ai/api/v1")
r = or_client.chat.completions.create(
    model=settings.visual_model,
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=500      # reasoning models need room to think before answering
)
msg = r.choices[0].message
content = msg.content or ""
reasoning = getattr(msg, "reasoning", "") or ""

if content.strip():
    print(f"  OpenRouter content: {content.strip()}")
elif reasoning.strip():
    print(f"  OpenRouter (reasoning only, no content yet): {reasoning[:80]}...")
    print("  WARNING: model returned reasoning but no content — increase max_tokens further")
else:
    print("  OpenRouter: empty response")

print("Testing Groq...")
groq_client = Groq(api_key=settings.groq_api_key)
r = groq_client.chat.completions.create(
    model=settings.synthesis_model,
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=5
)
print(f"  Groq: {r.choices[0].message.content.strip()}")

print("Testing Gemini...")
genai.configure(api_key=settings.google_api_key)
model = genai.GenerativeModel(settings.critic_model)
r = model.generate_content("Say OK")
print(f"  Gemini: {r.text.strip()}")

print("All APIs connected.")